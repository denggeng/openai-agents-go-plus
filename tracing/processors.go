// Copyright 2025 The NLP Odyssey Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tracing

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/openai/openai-go/v3/packages/param"
)

// ConsoleSpanExporter is an Exporter that prints the traces and spans to the console.
type ConsoleSpanExporter struct{}

func (c ConsoleSpanExporter) Export(items []any) error {
	for _, item := range items {
		switch v := item.(type) {
		case Trace:
			fmt.Printf("[Exporter] Export trace_id=%s, name=%s\n", v.TraceID(), v.Name())
		case Span:
			fmt.Printf("[Exporter] Export span: %+v", v.Export())
		default:
			return fmt.Errorf("ConsoleSpanExporter: unexpected item type %T", item)
		}
	}
	return nil
}

const DefaultBackendSpanExporterEndpoint = "https://api.openai.com/v1/traces/ingest"

type BackendSpanExporter struct {
	apiKey       atomic.Pointer[string]
	organization string
	project      string
	Endpoint     string
	MaxRetries   int
	BaseDelay    time.Duration
	MaxDelay     time.Duration
	client       *http.Client
}

type BackendSpanExporterParams struct {
	// The API key for the "Authorization" header
	// Defaults to OPENAI_API_KEY environment variable if not provided.
	APIKey string
	// The OpenAI organization to use.
	// Defaults to OPENAI_ORG_ID environment variable if not provided.
	Organization string
	// The OpenAI project to use.
	// Defaults to OPENAI_PROJECT_ID environment variable if not provided.
	Project string
	// The HTTP endpoint to which traces/spans are posted.
	// Defaults to DefaultBackendSpanExporterEndpoint if not provided.
	Endpoint string
	// Maximum number of retries upon failures.
	// Default: 3.
	MaxRetries param.Opt[int]
	// Base delay for the first backoff.
	// Default: 1 second.
	BaseDelay param.Opt[time.Duration]
	// Maximum delay for backoff growth.
	// Default: 30 seconds.
	MaxDelay param.Opt[time.Duration]
	// Optional custom http.Client.
	HTTPClient *http.Client
}

func NewBackendSpanExporter(params BackendSpanExporterParams) *BackendSpanExporter {
	b := &BackendSpanExporter{
		organization: params.Organization,
		project:      params.Project,
		Endpoint:     cmp.Or(params.Endpoint, DefaultBackendSpanExporterEndpoint),
		MaxRetries:   params.MaxRetries.Or(3),
		BaseDelay:    params.BaseDelay.Or(1 * time.Second),
		MaxDelay:     params.MaxDelay.Or(30 * time.Second),
		client:       cmp.Or(params.HTTPClient, &http.Client{Timeout: 60 * time.Second}),
	}
	if params.APIKey != "" {
		b.apiKey.Store(&params.APIKey)
	}
	return b
}

// SetAPIKey sets the OpenAI API key for the exporter.
func (b *BackendSpanExporter) SetAPIKey(apiKey string) {
	b.apiKey.Store(&apiKey)
}

func (b *BackendSpanExporter) APIKey() string {
	if v := b.apiKey.Load(); v != nil && *v != "" {
		return *v
	}
	return os.Getenv("OPENAI_API_KEY")
}

func (b *BackendSpanExporter) Organization() string {
	if b.organization == "" {
		return os.Getenv("OPENAI_ORG_ID")
	}
	return b.organization
}

func (b *BackendSpanExporter) Project() string {
	if b.project == "" {
		return os.Getenv("OPENAI_PROJECT_ID")
	}
	return b.project
}

type tracingAPIKeyProvider interface {
	TracingAPIKey() string
}

type exportableItem interface {
	Export() map[string]any
}

const maxOpenAITracingFieldBytes = 100 * 1024

func shouldSanitizeForOpenAITracingEndpoint(endpoint string) bool {
	normalized := strings.ToLower(strings.TrimSpace(endpoint))
	return strings.Contains(normalized, "api.openai.com") && strings.Contains(normalized, "/v1/traces/ingest")
}

func sanitizeTracingItems(items []map[string]any) []map[string]any {
	if len(items) == 0 {
		return nil
	}
	sanitized := make([]map[string]any, 0, len(items))
	for _, item := range items {
		sanitized = append(sanitized, sanitizeTracingItem(item))
	}
	return sanitized
}

func sanitizeTracingItem(item map[string]any) map[string]any {
	sanitizedAny := sanitizeTracingValue(item, make(map[uintptr]struct{}))
	sanitized, ok := sanitizedAny.(map[string]any)
	if !ok {
		return map[string]any{}
	}
	if spanData, ok := sanitized["span_data"].(map[string]any); ok {
		if input, ok := spanData["input"]; ok {
			spanData["input"] = truncateTracingField(input, maxOpenAITracingFieldBytes)
		}
		if output, ok := spanData["output"]; ok {
			spanData["output"] = truncateTracingField(output, maxOpenAITracingFieldBytes)
		}
		if usage, ok := spanData["usage"].(map[string]any); ok {
			spanData["usage"] = normalizeTracingUsage(usage)
		}
		sanitized["span_data"] = spanData
	}
	return sanitized
}

func normalizeTracingUsage(usage map[string]any) map[string]any {
	if len(usage) == 0 {
		return nil
	}
	out := make(map[string]any, 2)
	details := make(map[string]any)
	for key, value := range usage {
		switch key {
		case "input_tokens", "output_tokens":
			out[key] = value
		default:
			details[key] = value
		}
	}
	if len(details) > 0 {
		out["details"] = details
	}
	return out
}

func truncateTracingField(value any, maxBytes int) any {
	data, err := json.Marshal(value)
	if err != nil || len(data) <= maxBytes {
		return value
	}
	if asString, ok := value.(string); ok {
		return truncateStringByBytes(asString, maxBytes)
	}
	preview := string(data[:maxBytes])
	return map[string]any{
		"truncated":      true,
		"original_bytes": len(data),
		"preview":        preview,
	}
}

func truncateStringByBytes(value string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(value) <= maxBytes {
		return value
	}

	byteLen := 0
	end := 0
	for i, r := range value {
		runeLen := utf8.RuneLen(r)
		if runeLen < 0 {
			runeLen = 1
		}
		if byteLen+runeLen > maxBytes {
			break
		}
		byteLen += runeLen
		end = i + runeLen
	}
	return value[:end]
}

func sanitizeTracingValue(value any, seen map[uintptr]struct{}) any {
	if value == nil {
		return nil
	}
	v := reflect.ValueOf(value)
	if !v.IsValid() {
		return nil
	}
	switch v.Kind() {
	case reflect.Float32, reflect.Float64:
		f := v.Convert(reflect.TypeOf(float64(0))).Float()
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return fmt.Sprintf("%v", f)
		}
		return value
	case reflect.Map:
		if v.IsNil() {
			return nil
		}
		ptr := v.Pointer()
		if ptr != 0 {
			if _, exists := seen[ptr]; exists {
				return "<cyclic>"
			}
			seen[ptr] = struct{}{}
			defer delete(seen, ptr)
		}
		out := make(map[string]any, v.Len())
		iter := v.MapRange()
		for iter.Next() {
			key := fmt.Sprint(iter.Key().Interface())
			out[key] = sanitizeTracingValue(iter.Value().Interface(), seen)
		}
		return out
	case reflect.Slice:
		if v.IsNil() {
			return nil
		}
		ptr := v.Pointer()
		if ptr != 0 {
			if _, exists := seen[ptr]; exists {
				return "<cyclic>"
			}
			seen[ptr] = struct{}{}
			defer delete(seen, ptr)
		}
		fallthrough
	case reflect.Array:
		out := make([]any, v.Len())
		for i := 0; i < v.Len(); i++ {
			out[i] = sanitizeTracingValue(v.Index(i).Interface(), seen)
		}
		return out
	case reflect.Pointer:
		if v.IsNil() {
			return nil
		}
		ptr := v.Pointer()
		if ptr != 0 {
			if _, exists := seen[ptr]; exists {
				return "<cyclic>"
			}
			seen[ptr] = struct{}{}
			defer delete(seen, ptr)
		}
		return sanitizeTracingValue(v.Elem().Interface(), seen)
	case reflect.Interface:
		if v.IsNil() {
			return nil
		}
		return sanitizeTracingValue(v.Elem().Interface(), seen)
	default:
		return value
	}
}

func (b *BackendSpanExporter) Export(ctx context.Context, items []any) error {
	if len(items) == 0 {
		return nil
	}

	grouped := make(map[string][]map[string]any)
	for _, item := range items {
		exportable, ok := item.(exportableItem)
		if !ok {
			switch v := item.(type) {
			case Trace:
				exportable = v
			case Span:
				exportable = v
			default:
				return fmt.Errorf("BackendSpanExporter: unexpected item type %T", item)
			}
		}
		exported := exportable.Export()
		if exported == nil {
			continue
		}
		apiKey := ""
		if provider, ok := item.(tracingAPIKeyProvider); ok {
			apiKey = provider.TracingAPIKey()
		}
		grouped[apiKey] = append(grouped[apiKey], exported)
	}

	for itemKey, data := range grouped {
		apiKey := itemKey
		if apiKey == "" {
			apiKey = b.APIKey()
		}
		if apiKey == "" {
			Logger().Warn("BackendSpanExporter: OpenAI API key is not set, skipping trace export")
			continue
		}

		payloadData := data
		if shouldSanitizeForOpenAITracingEndpoint(b.Endpoint) {
			payloadData = sanitizeTracingItems(payloadData)
		}
		payload := map[string]any{
			"data": payloadData,
		}

		header := make(http.Header)
		header.Set("Authorization", "Bearer "+apiKey)
		header.Set("Content-Type", "application/json")
		header.Set("OpenAI-Beta", "traces=v1")
		if b.Organization() != "" {
			header.Set("OpenAI-Organization", b.Organization())
		}
		if b.Project() != "" {
			header.Set("OpenAI-Project", b.Project())
		}

		jsonPayload, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("failed to JSON-marshal tracing payload: %w", err)
		}

		// Exponential backoff loop
		attempt := 0
		delay := b.BaseDelay
		for {
			attempt += 1

			request, err := http.NewRequestWithContext(ctx, http.MethodPost, b.Endpoint, bytes.NewReader(jsonPayload))
			if err != nil {
				return fmt.Errorf("failed to initialize new tracing request: %w", err)
			}
			request.Header = header

			response, err := b.client.Do(request)

			if err != nil {
				Logger().Warn("[non-fatal] Tracing: request failed", slog.String("error", err.Error()))
			} else {
				// If the response is successful, break out of the loop
				if response.StatusCode < 300 {
					_ = response.Body.Close()
					Logger().Debug(fmt.Sprintf("Exported %d items", len(payloadData)))
					break
				}

				// If the response is a client error (4xx), we won't retry
				if response.StatusCode >= 400 && response.StatusCode < 500 {
					body, err := io.ReadAll(response.Body)
					if err != nil {
						Logger().Warn("failed to read tracing response body", slog.String("error", err.Error()))
					}
					_ = response.Body.Close()
					Logger().Warn(
						"[non-fatal] Tracing client error",
						slog.Int("statusCode", response.StatusCode),
						slog.String("response", string(body)),
					)
					break
				}
				_ = response.Body.Close()

				// For 5xx or other unexpected codes, treat it as transient and retry
				Logger().Warn("[non-fatal] Tracing: server error, retrying.", slog.Int("statusCode", response.StatusCode))
			}

			//# If we reach here, we need to retry or give up
			if attempt >= b.MaxRetries {
				Logger().Error("[non-fatal] Tracing: max retries reached, giving up on this batch.")
				break
			}

			// Exponential backoff + jitter
			sleepTime := delay + time.Duration(rand.Int64N(int64(delay/10))) // 10% jitter
			time.Sleep(sleepTime)
			delay = min(delay*2, b.MaxDelay)
		}
	}
	return nil
}

// Close the underlying HTTP client's idle connections.
func (b *BackendSpanExporter) Close() {
	b.client.CloseIdleConnections()
}

type BatchTraceProcessor struct {
	exporter      Exporter
	maxQueueSize  int
	maxBatchSize  int
	scheduleDelay time.Duration
	// The queue size threshold at which we export immediately.
	exportTriggerSize int
	// Track when we next *must* perform a scheduled export.
	nextExportTime time.Time
	shutdownCalled atomic.Bool
	workerRunning  atomic.Bool
	workerDoneChan chan struct{}
	workerMu       sync.RWMutex

	queueMu   sync.RWMutex
	queueChan chan any
	queueSize int
}

type BatchTraceProcessorParams struct {
	// The exporter to use.
	Exporter Exporter
	// The maximum number of spans to store in the queue.
	// After this, we will start dropping spans.
	// Default: 8192.
	MaxQueueSize param.Opt[int]
	// The maximum number of spans to export in a single batch.
	// Default: 128.
	MaxBatchSize param.Opt[int]
	// The delay between checks for new spans to export.
	// Default: 5 seconds.
	ScheduleDelay param.Opt[time.Duration]
	// The ratio of the queue size at which we will trigger an export.
	// Default: 0.7.
	ExportTriggerRatio param.Opt[float64]
}

func NewBatchTraceProcessor(params BatchTraceProcessorParams) *BatchTraceProcessor {
	maxQueueSize := params.MaxQueueSize.Or(8192)
	scheduleDelay := params.ScheduleDelay.Or(5 * time.Second)
	exportTriggerRatio := params.ExportTriggerRatio.Or(0.7)

	return &BatchTraceProcessor{
		exporter:          params.Exporter,
		maxQueueSize:      maxQueueSize,
		maxBatchSize:      params.MaxBatchSize.Or(128),
		scheduleDelay:     scheduleDelay,
		exportTriggerSize: max(1, int(float64(maxQueueSize)*exportTriggerRatio)),
		nextExportTime:    time.Now().Add(scheduleDelay),
		queueChan:         make(chan any, maxQueueSize),
		queueSize:         0,
	}
}

func (b *BatchTraceProcessor) OnTraceStart(ctx context.Context, trace Trace) error {
	// Ensure the background worker is running before we enqueue anything.
	b.ensureWorkerStarted(ctx)

	b.queueMu.Lock()
	defer b.queueMu.Unlock()

	select {
	case b.queueChan <- trace:
		b.queueSize += 1
	default:
		Logger().Warn("Queue is full, dropping trace.")
	}
	return nil
}

func (b *BatchTraceProcessor) OnTraceEnd(ctx context.Context, trace Trace) error {
	// We send traces via OnTraceStart, so we don't need to do anything here.
	return nil
}

func (b *BatchTraceProcessor) OnSpanStart(ctx context.Context, span Span) error {
	// We send spans via OnSpanEnd, so we don't need to do anything here.
	return nil
}

func (b *BatchTraceProcessor) OnSpanEnd(ctx context.Context, span Span) error {
	// Ensure the background worker is running before we enqueue anything.
	b.ensureWorkerStarted(ctx)

	b.queueMu.Lock()
	defer b.queueMu.Unlock()

	select {
	case b.queueChan <- span:
		b.queueSize += 1
	default:
		Logger().Warn("Queue is full, dropping span.")
	}
	return nil
}

// Shutdown is called when the application stops.
// We signal our worker goroutine to stop, then wait for its completion.
func (b *BatchTraceProcessor) Shutdown(ctx context.Context) error {
	b.shutdownCalled.Store(true)

	// Only wait if we ever started the background worker; otherwise flush synchronously.
	if b.workerRunning.Load() {
		<-b.workerDoneChan
		return nil
	}

	// No background goroutine: process any remaining items synchronously.
	return b.exportBatches(ctx, true)
}

// ForceFlush forces an immediate flush of all queued spans.
func (b *BatchTraceProcessor) ForceFlush(ctx context.Context) error {
	return b.exportBatches(ctx, true)
}

func (b *BatchTraceProcessor) ensureWorkerStarted(ctx context.Context) {
	// Fast path without holding the lock.
	if b.workerRunning.Load() {
		return
	}

	b.workerMu.Lock()
	defer b.workerMu.Unlock()
	if b.workerRunning.Load() {
		return
	}

	b.workerDoneChan = make(chan struct{})
	b.workerRunning.Store(true)

	go func() {
		defer func() {
			b.workerMu.Lock()
			defer b.workerMu.Unlock()
			b.workerRunning.Store(false)
			close(b.workerDoneChan)
		}()

		err := b.run(ctx)
		if err != nil {
			Logger().Error("BatchTraceProcessor worker error", slog.String("error", err.Error()))
		}
	}()
}

func (b *BatchTraceProcessor) run(ctx context.Context) error {
	for !b.shutdownCalled.Load() {
		currentTime := time.Now()

		b.queueMu.RLock()
		queueSize := b.queueSize
		b.queueMu.RUnlock()

		// TODO: this could be improved using sync.Cond, avoiding sleep

		// If it's time for a scheduled flush or queue is above the trigger threshold
		if currentTime.After(b.nextExportTime) || queueSize >= b.exportTriggerSize {
			err := b.exportBatches(ctx, false)
			if err != nil {
				return err
			}
			// Reset the next scheduled flush time
			b.nextExportTime = time.Now().Add(b.scheduleDelay)
		} else {
			// Sleep a short interval so we don't busy-wait.
			time.Sleep(200 * time.Millisecond)
		}
	}

	// Final drain after shutdown
	return b.exportBatches(ctx, true)
}

// exportBatches drains the queue and exports in batches. If force=true, export everything.
// Otherwise, export up to `maxBatchSize` repeatedly until the queue is completely empty.
func (b *BatchTraceProcessor) exportBatches(ctx context.Context, force bool) error {
	for {
		var itemsToExport []any

		// Gather a batch of spans up to maxBatchSize
	queueLoop:
		for {
			b.queueMu.Lock()
			queueSize := b.queueSize
			if !(queueSize > 0 && (force || len(itemsToExport) < b.maxBatchSize)) {
				b.queueMu.Unlock()
				break queueLoop
			}

			select {
			case item := <-b.queueChan:
				b.queueSize -= 1
				b.queueMu.Unlock()
				itemsToExport = append(itemsToExport, item)
			default:
				b.queueMu.Unlock()
				// Another goroutine might have emptied the queue between checks
				break queueLoop
			}
		}

		// If we collected nothing, we're done
		if len(itemsToExport) == 0 {
			break
		}

		// Export the batch
		err := b.exporter.Export(ctx, itemsToExport)
		if err != nil {
			return err
		}
	}

	return nil
}

var globalExporter atomic.Pointer[BackendSpanExporter]
var globalProcessor atomic.Pointer[BatchTraceProcessor]
var defaultExporterOnce sync.Once
var defaultProcessorOnce sync.Once

// DefaultExporter returns the default exporter, which exports traces and
// spans to the backend in batches.
func DefaultExporter() *BackendSpanExporter {
	if exporter := globalExporter.Load(); exporter != nil {
		return exporter
	}
	defaultExporterOnce.Do(func() {
		if globalExporter.Load() != nil {
			return
		}
		exporter := NewBackendSpanExporter(BackendSpanExporterParams{})
		globalExporter.Store(exporter)
	})
	return globalExporter.Load()
}

// DefaultProcessor returns the default processor, which exports traces and
// spans to the backend in batches.
func DefaultProcessor() *BatchTraceProcessor {
	if processor := globalProcessor.Load(); processor != nil {
		return processor
	}
	defaultProcessorOnce.Do(func() {
		if globalProcessor.Load() != nil {
			return
		}
		processor := NewBatchTraceProcessor(BatchTraceProcessorParams{
			Exporter: DefaultExporter(),
		})
		globalProcessor.Store(processor)
	})
	return globalProcessor.Load()
}
