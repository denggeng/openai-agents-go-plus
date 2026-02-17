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

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"

	realtime "github.com/denggeng/openai-agents-go-plus/agents/realtime"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/webhooks"
)

type callTask interface {
	Done() bool
}

type backgroundTask struct {
	done chan struct{}
}

func (t *backgroundTask) Done() bool {
	select {
	case <-t.done:
		return true
	default:
		return false
	}
}

var (
	activeCallTasks   = map[string]callTask{}
	activeCallTasksMu sync.Mutex
	startObserver     = defaultStartObserver
)

func defaultStartObserver(callID string) callTask {
	task := &backgroundTask{done: make(chan struct{})}
	go func() {
		defer close(task.done)
		observeCall(callID)
		clearCallTask(callID, task)
	}()
	return task
}

func trackCallTask(callID string) {
	activeCallTasksMu.Lock()
	defer activeCallTasksMu.Unlock()

	if existing, ok := activeCallTasks[callID]; ok {
		if !existing.Done() {
			log.Printf("call %s already has an active observer; ignoring duplicate webhook delivery.", callID)
			return
		}
		delete(activeCallTasks, callID)
	}

	activeCallTasks[callID] = startObserver(callID)
}

func clearCallTask(callID string, task callTask) {
	activeCallTasksMu.Lock()
	defer activeCallTasksMu.Unlock()
	if current, ok := activeCallTasks[callID]; ok && current == task {
		delete(activeCallTasks, callID)
	}
}

type twilioServer struct {
	client        openai.Client
	startingAgent *realtime.RealtimeAgent[any]
}

func (s *twilioServer) handleWebhook(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("failed to read request body"))
		return
	}
	defer r.Body.Close()

	webhookEvent, err := s.client.Webhooks.Unwrap(body, r.Header)
	if err != nil {
		log.Printf("invalid webhook signature: %v", err)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("invalid webhook signature"))
		return
	}

	switch event := webhookEvent.AsAny().(type) {
	case webhooks.RealtimeCallIncomingWebhookEvent:
		callID := event.Data.CallID
		instructions := resolveAgentInstructions(s.startingAgent)
		if err := acceptCall(r.Context(), s.client, callID, instructions); err != nil {
			var apiErr *openai.Error
			if errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusNotFound {
				log.Printf("call %s no longer exists when attempting accept (404).", callID)
				w.WriteHeader(http.StatusOK)
				return
			}
			log.Printf("failed to accept call %s: %v", callID, err)
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte("failed to accept call"))
			return
		}
		trackCallTask(callID)
		w.WriteHeader(http.StatusOK)
		return
	default:
		w.WriteHeader(http.StatusOK)
		return
	}
}

func (s *twilioServer) handleHealthcheck(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func acceptCall(ctx context.Context, client openai.Client, callID string, instructions string) error {
	payload := map[string]any{
		"type":         "realtime",
		"model":        "gpt-realtime",
		"instructions": instructions,
	}
	return client.Post(ctx, fmt.Sprintf("/realtime/calls/%s/accept", callID), payload, nil)
}

func resolveAgentInstructions(agent *realtime.RealtimeAgent[any]) string {
	if agent == nil {
		return "You are a helpful triage agent for ABC customer service."
	}
	if text, ok := agent.Instructions.(string); ok && strings.TrimSpace(text) != "" {
		return text
	}
	return "You are a helpful triage agent for ABC customer service."
}

func observeCall(callID string) {
	ctx := context.Background()
	agent := getStartingAgent()
	runner := realtime.NewRealtimeRunner(agent, realtime.NewOpenAIRealtimeSIPModel(), nil)

	modelConfig := &realtime.RealtimeModelConfig{
		CallID: callID,
		InitialSettings: realtime.RealtimeSessionModelSettings{
			"turn_detection": map[string]any{
				"type":               "semantic_vad",
				"interrupt_response": true,
			},
		},
	}
	if err := runRealtimeSession(ctx, runner, modelConfig); err != nil {
		log.Printf("error while observing call %s: %v", callID, err)
	}
}

func runRealtimeSession(
	ctx context.Context,
	runner *realtime.RealtimeRunner,
	modelConfig *realtime.RealtimeModelConfig,
) error {
	session := runner.Run(nil, modelConfig)
	if err := session.Enter(ctx); err != nil {
		return err
	}
	defer func() {
		_ = session.Close(ctx)
	}()

	greeting := fmt.Sprintf(
		"Say exactly '%s' now before continuing the conversation.",
		welcomeMessage,
	)
	_ = session.Model().SendEvent(ctx, realtime.RealtimeModelSendRawMessage{
		Message: realtime.RealtimeModelRawClientMessage{
			Type: "response.create",
			OtherData: map[string]any{
				"response": map[string]any{
					"instructions": greeting,
				},
			},
		},
	})

	for {
		select {
		case event, ok := <-session.Events():
			if !ok {
				return nil
			}
			if shouldStop, err := handleSessionEvent(event); shouldStop {
				return err
			}
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func handleSessionEvent(event realtime.RealtimeSessionEvent) (bool, error) {
	switch e := event.(type) {
	case realtime.RealtimeHistoryAddedEvent:
		logHistoryItem(e.Item)
	case realtime.RealtimeErrorEvent:
		log.Printf("realtime session error: %v", e.Error)
	case realtime.RealtimeRawModelEvent:
		if status, ok := e.Data.(realtime.RealtimeModelConnectionStatusEvent); ok {
			if status.Status == realtime.RealtimeConnectionStatusDisconnected {
				return true, nil
			}
		}
	}
	return false, nil
}

func logHistoryItem(item any) {
	mapping, ok := item.(map[string]any)
	if !ok {
		return
	}
	role, _ := mapping["role"].(string)
	content, _ := mapping["content"].([]any)
	for _, part := range content {
		partMap, ok := part.(map[string]any)
		if !ok {
			continue
		}
		partType, _ := partMap["type"].(string)
		switch role {
		case "user":
			if partType == "input_text" {
				if text, ok := partMap["text"].(string); ok && strings.TrimSpace(text) != "" {
					log.Printf("Caller: %s", text)
				}
			}
		case "assistant":
			switch partType {
			case "text", "output_text":
				if text, ok := partMap["text"].(string); ok && strings.TrimSpace(text) != "" {
					log.Printf("Assistant (text): %s", text)
				}
			case "audio", "output_audio":
				if transcript, ok := partMap["transcript"].(string); ok && strings.TrimSpace(transcript) != "" {
					log.Printf("Assistant (audio transcript): %s", transcript)
				}
			}
		}
	}
}

func mustGetEnv(name string) string {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		log.Fatalf("missing environment variable: %s", name)
	}
	return value
}

func main() {
	apiKey := mustGetEnv("OPENAI_API_KEY")
	webhookSecret := mustGetEnv("OPENAI_WEBHOOK_SECRET")
	port := strings.TrimSpace(os.Getenv("PORT"))
	if port == "" {
		port = "8080"
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithWebhookSecret(webhookSecret),
	)

	server := &twilioServer{
		client:        client,
		startingAgent: getStartingAgent(),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/openai/webhook", server.handleWebhook)
	mux.HandleFunc("/", server.handleHealthcheck)

	log.Printf("twilio SIP server listening on :%s", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal(err)
	}
}
