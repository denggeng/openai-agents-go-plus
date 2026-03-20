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

package agents

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// MCPServer is implemented by Model Context Protocol servers.
type MCPServer interface {
	// Connect to the server.
	//
	// For example, this might mean spawning a subprocess or opening a network connection.
	// The server is expected to remain connected until Cleanup is called.
	Connect(context.Context) error

	// Cleanup the server.
	//
	// For example, this might mean closing a subprocess or closing a network connection.
	Cleanup(context.Context) error

	// Name returns a readable name for the server.
	Name() string

	// UseStructuredContent reports  whether to use a tool result's
	// `StructuredContent` when calling an MCP tool.
	UseStructuredContent() bool

	// ListTools lists the tools available on the server.
	ListTools(context.Context, *Agent) ([]*mcp.Tool, error)

	// CallTool invokes a tool on the server.
	CallTool(ctx context.Context, toolName string, arguments map[string]any, meta map[string]any) (*mcp.CallToolResult, error)

	// ListPrompts lists the prompts available on the server.
	ListPrompts(ctx context.Context) (*mcp.ListPromptsResult, error)

	// GetPrompt returns a specific prompt from the server.
	GetPrompt(ctx context.Context, name string, arguments map[string]string) (*mcp.GetPromptResult, error)
}

type mcpClientSession interface {
	ListTools(context.Context, *mcp.ListToolsParams) (*mcp.ListToolsResult, error)
	CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error)
	ListPrompts(context.Context, *mcp.ListPromptsParams) (*mcp.ListPromptsResult, error)
	GetPrompt(context.Context, *mcp.GetPromptParams) (*mcp.GetPromptResult, error)
	Close() error
}

var newMCPClient = mcp.NewClient

// MCPRequestAuthenticator mutates an outgoing MCP HTTP request before it is sent.
type MCPRequestAuthenticator func(*http.Request) error

func newRequestSerializationSlot() chan struct{} {
	slot := make(chan struct{}, 1)
	slot <- struct{}{}
	return slot
}

// MCPServerWithClientSession is a base type for MCP servers that uses an
// mcp.ClientSession to communicate with the server.
type MCPServerWithClientSession struct {
	transport                mcp.Transport
	session                  mcpClientSession
	cleanupMu                sync.Mutex
	cacheToolsList           bool
	cacheDirty               bool
	toolsList                []*mcp.Tool
	toolFilter               MCPToolFilter
	name                     string
	useStructuredContent     bool
	maxRetryAttempts         int
	retryBackoffBase         time.Duration
	clientOptions            *mcp.ClientOptions
	clientSessionTimeout     time.Duration
	needsApprovalPolicy      mcpNeedsApprovalPolicy
	failureErrorFunction     *ToolErrorFunction
	failureErrorSet          bool
	toolMetaResolver         MCPToolMetaResolver
	requestSerializationSlot chan struct{}
	cleanupHook              func()
}

type MCPServerWithClientSessionParams struct {
	Name      string
	Transport mcp.Transport
	// Optional client options, including MCP message handlers.
	ClientOptions *mcp.ClientOptions

	// Optional per-request timeout used for MCP client session calls.
	// Applies to ListTools/CallTool/ListPrompts/GetPrompt.
	ClientSessionTimeout time.Duration

	// Whether to cache the tools list. If `true`, the tools list will be
	// cached and only fetched from the server once. If `false`, the tools list will be
	// fetched from the server on each call to `ListTools()`. The cache can be invalidated
	// by calling `InvalidateToolsCache()`. You should set this to `true` if you know the
	// server will not change its tools list, because it can drastically improve latency
	// (by avoiding a round-trip to the server every time).
	CacheToolsList bool

	// The tool filter to use for filtering tools.
	ToolFilter MCPToolFilter

	// Whether to use `StructuredContent` when calling an MCP tool.
	// Defaults to false for backwards compatibility - most MCP servers still include
	// the structured content in `Content`, and using it by default will cause duplicate
	// content. You can set this to true if you know the server will not duplicate
	// the structured content in `Content`.
	UseStructuredContent bool

	// Maximum number of retry attempts for ListTools/CallTool. Use -1 for unlimited retries.
	MaxRetryAttempts int

	// Base delay for exponential backoff between retries.
	RetryBackoffBase time.Duration

	// Optional approval policy for tools in this MCP server.
	// Supported forms:
	// - bool
	// - "always"/"never"
	// - map[string]bool
	// - map[string]string where values are "always"/"never"
	// - MCPRequireApprovalObject
	// - map[string]any using TS-style {always:{tool_names:[...]}, never:{tool_names:[...]}}
	RequireApproval any

	// Optional resolver for MCP request metadata (`_meta`) on tool calls.
	ToolMetaResolver MCPToolMetaResolver

	// Optional per-server override for MCP tool failure handling.
	// Set FailureErrorFunctionSet=true with nil FailureErrorFunction to re-raise errors.
	FailureErrorFunction    *ToolErrorFunction
	FailureErrorFunctionSet bool
}

func NewMCPServerWithClientSession(params MCPServerWithClientSessionParams) *MCPServerWithClientSession {
	return &MCPServerWithClientSession{
		transport:      params.Transport,
		cacheToolsList: params.CacheToolsList,
		// The cache is always dirty at startup, so that we fetch tools at least once
		cacheDirty:           true,
		toolFilter:           params.ToolFilter,
		name:                 params.Name,
		useStructuredContent: params.UseStructuredContent,
		maxRetryAttempts:     params.MaxRetryAttempts,
		retryBackoffBase:     params.RetryBackoffBase,
		clientOptions:        params.ClientOptions,
		clientSessionTimeout: params.ClientSessionTimeout,
		needsApprovalPolicy:  normalizeMCPNeedsApprovalPolicy(params.RequireApproval),
		toolMetaResolver:     params.ToolMetaResolver,
		failureErrorFunction: cloneToolErrorFunctionPointer(params.FailureErrorFunction),
		failureErrorSet:      params.FailureErrorFunctionSet,
	}
}

func (s *MCPServerWithClientSession) Connect(ctx context.Context) (err error) {
	defer func() {
		if err != nil {
			Logger().Error("Error initializing MCP server", slog.String("error", err.Error()))
			if e := s.Cleanup(ctx); e != nil {
				err = errors.Join(err, fmt.Errorf("MCP server cleanup error: %w", e))
			}
		}
	}()

	client := newMCPClient(&mcp.Implementation{Name: s.name}, s.clientOptions)
	session, err := client.Connect(ctx, s.transport, nil)
	if err != nil {
		return fmt.Errorf("MCP client connection error: %w", err)
	}
	s.session = session
	return nil
}

func (s *MCPServerWithClientSession) Cleanup(context.Context) error {
	s.cleanupMu.Lock()
	defer func() {
		s.session = nil
		if s.cleanupHook != nil {
			s.cleanupHook()
		}
		s.cleanupMu.Unlock()
	}()

	if s.session != nil {
		err := s.session.Close()
		if err != nil {
			Logger().Error("Error cleaning up server", slog.String("error", err.Error()))
		}
		return err
	}
	return nil
}

func (s *MCPServerWithClientSession) Name() string {
	return s.name
}

func (s *MCPServerWithClientSession) UseStructuredContent() bool {
	return s.useStructuredContent
}

func (s *MCPServerWithClientSession) ListTools(ctx context.Context, agent *Agent) ([]*mcp.Tool, error) {
	if s.session == nil {
		return nil, NewUserError("server not initialized: make sure you call `Connect()` first")
	}

	var tools []*mcp.Tool
	// Return from cache if caching is enabled, we have tools, and the cache is not dirty
	if s.cacheToolsList && !s.cacheDirty && len(s.toolsList) > 0 {
		tools = s.toolsList
	} else {
		s.cacheDirty = false
		var listToolsResults *mcp.ListToolsResult
		err := s.runWithRetries(ctx, func() error {
			var err error
			listToolsResults, err = withSerializedSessionRequest(
				ctx,
				s.requestSerializationSlot,
				func() (*mcp.ListToolsResult, error) {
					attemptCtx, cancel := s.withSessionTimeout(ctx)
					defer cancel()
					return s.session.ListTools(attemptCtx, nil)
				},
			)
			return err
		})
		if err != nil {
			return nil, fmt.Errorf("MCP list tools error: %w", err)
		}
		tools = listToolsResults.Tools
		s.toolsList = tools
	}

	filteredTools := tools
	if s.toolFilter != nil {
		if agent == nil {
			return nil, UserErrorf("agent is required for dynamic tool filtering")
		}
		filterContext := MCPToolFilterContext{
			Agent:      agent,
			ServerName: s.name,
		}
		filteredTools = ApplyMCPToolFilter(ctx, filterContext, s.toolFilter, filteredTools, agent)
	}
	return filteredTools, nil
}

func (s *MCPServerWithClientSession) CallTool(
	ctx context.Context,
	toolName string,
	arguments map[string]any,
	meta map[string]any,
) (*mcp.CallToolResult, error) {
	if s.session == nil {
		return nil, NewUserError("server not initialized: make sure you call `Connect()` first")
	}
	if err := s.validateCallToolArguments(toolName, arguments); err != nil {
		return nil, err
	}
	var result *mcp.CallToolResult
	err := s.runWithRetries(ctx, func() error {
		var err error
		result, err = withSerializedSessionRequest(
			ctx,
			s.requestSerializationSlot,
			func() (*mcp.CallToolResult, error) {
				attemptCtx, cancel := s.withSessionTimeout(ctx)
				defer cancel()

				params := &mcp.CallToolParams{
					Name:      toolName,
					Arguments: arguments,
				}
				if len(meta) > 0 {
					params.Meta = mcp.Meta(meta)
				}
				return s.session.CallTool(attemptCtx, params)
			},
		)
		return err
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

func (s *MCPServerWithClientSession) validateCallToolArguments(toolName string, arguments map[string]any) error {
	tool := s.findCachedTool(toolName)
	if tool == nil || tool.InputSchema == nil {
		return nil
	}

	required := tool.InputSchema.Required
	if len(required) == 0 {
		return nil
	}
	if arguments == nil {
		return UserErrorf("missing required parameters: %s", strings.Join(required, ", "))
	}
	var missing []string
	for _, key := range required {
		if _, ok := arguments[key]; !ok {
			missing = append(missing, key)
		}
	}
	if len(missing) > 0 {
		return UserErrorf("missing required parameters: %s", strings.Join(missing, ", "))
	}
	return nil
}

func (s *MCPServerWithClientSession) findCachedTool(toolName string) *mcp.Tool {
	if len(s.toolsList) == 0 {
		return nil
	}
	for _, tool := range s.toolsList {
		if tool != nil && tool.Name == toolName {
			return tool
		}
	}
	return nil
}

func (s *MCPServerWithClientSession) runWithRetries(
	ctx context.Context,
	fn func() error,
) error {
	if ctx == nil {
		ctx = context.Background()
	}
	attempts := 0
	for {
		err := fn()
		if err == nil {
			return nil
		}
		if ctx != nil && ctx.Err() != nil {
			return ctx.Err()
		}
		attempts++
		if s.maxRetryAttempts != -1 && attempts > s.maxRetryAttempts {
			return err
		}
		if err := s.sleepRetryBackoff(ctx, attempts-1); err != nil {
			return err
		}
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (s *MCPServerWithClientSession) sleepRetryBackoff(ctx context.Context, attemptIndex int) error {
	if s.retryBackoffBase <= 0 {
		return nil
	}
	backoff := s.retryBackoffBase * time.Duration(1<<maxInt(attemptIndex, 0))
	timer := time.NewTimer(backoff)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func withSerializedSessionRequest[T any](
	ctx context.Context,
	slot chan struct{},
	fn func() (T, error),
) (T, error) {
	var zero T
	if slot == nil {
		return fn()
	}
	if ctx == nil {
		ctx = context.Background()
	}

	select {
	case <-ctx.Done():
		return zero, ctx.Err()
	case <-slot:
	}
	defer func() { slot <- struct{}{} }()
	return fn()
}

func (s *MCPServerWithClientSession) withSessionTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s.clientSessionTimeout <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, s.clientSessionTimeout)
}

// CachedTools returns the latest cached tool list, if present.
func (s *MCPServerWithClientSession) CachedTools() []*mcp.Tool {
	if len(s.toolsList) == 0 {
		return nil
	}
	return slices.Clone(s.toolsList)
}

// MCPNeedsApprovalForTool returns the approval policy for a specific MCP tool.
func (s *MCPServerWithClientSession) MCPNeedsApprovalForTool(tool *mcp.Tool, agent *Agent) FunctionToolNeedsApproval {
	return s.needsApprovalPolicy.forTool(tool, agent)
}

// MCPFailureErrorFunctionOverride reports whether this server overrides MCP failure handling.
func (s *MCPServerWithClientSession) MCPFailureErrorFunctionOverride() (bool, *ToolErrorFunction) {
	if !s.failureErrorSet {
		return false, nil
	}
	if s.failureErrorFunction == nil {
		var nilFn ToolErrorFunction
		return true, &nilFn
	}
	return true, cloneToolErrorFunctionPointer(s.failureErrorFunction)
}

// MCPResolveToolMeta resolves `_meta` for an MCP tool call.
func (s *MCPServerWithClientSession) MCPResolveToolMeta(
	ctx context.Context,
	metaContext MCPToolMetaContext,
) (map[string]any, error) {
	if s.toolMetaResolver == nil {
		return nil, nil
	}
	// Resolver receives a deep copy so it cannot mutate invocation arguments.
	metaContext.Arguments = deepCopyMap(metaContext.Arguments)
	resolved, err := s.toolMetaResolver(ctx, metaContext)
	if err != nil {
		return nil, err
	}
	if resolved == nil {
		return nil, nil
	}
	return deepCopyMap(resolved), nil
}

func (s *MCPServerWithClientSession) ListPrompts(ctx context.Context) (*mcp.ListPromptsResult, error) {
	if s.session == nil {
		return nil, NewUserError("server not initialized: make sure you call `Connect()` first")
	}
	return withSerializedSessionRequest(
		ctx,
		s.requestSerializationSlot,
		func() (*mcp.ListPromptsResult, error) {
			attemptCtx, cancel := s.withSessionTimeout(ctx)
			defer cancel()
			return s.session.ListPrompts(attemptCtx, nil)
		},
	)
}

func (s *MCPServerWithClientSession) GetPrompt(ctx context.Context, name string, arguments map[string]string) (*mcp.GetPromptResult, error) {
	if s.session == nil {
		return nil, NewUserError("server not initialized: make sure you call `Connect()` first")
	}
	return withSerializedSessionRequest(
		ctx,
		s.requestSerializationSlot,
		func() (*mcp.GetPromptResult, error) {
			attemptCtx, cancel := s.withSessionTimeout(ctx)
			defer cancel()
			return s.session.GetPrompt(attemptCtx, &mcp.GetPromptParams{
				Name:      name,
				Arguments: arguments,
			})
		},
	)
}

func (s *MCPServerWithClientSession) Run(ctx context.Context, fn func(context.Context, *MCPServerWithClientSession) error) (err error) {
	err = s.Connect(ctx)
	if err != nil {
		return fmt.Errorf("MCP server connection error: %w", err)
	}
	defer func() {
		if e := s.Cleanup(ctx); e != nil {
			err = errors.Join(err, fmt.Errorf("MCP server cleanup error: %w", e))
		}
	}()
	return fn(ctx, s)
}

// InvalidateToolsCache invalidates the tools cache.
func (s *MCPServerWithClientSession) InvalidateToolsCache() {
	s.cacheDirty = true
}

type MCPServerStdioParams struct {
	// The command to run to start the server.
	Command *exec.Cmd

	// Optional client options, including MCP message handlers.
	ClientOptions *mcp.ClientOptions

	// Optional per-request timeout used for MCP client session calls.
	ClientSessionTimeout time.Duration

	// Whether to cache the tools list. If `true`, the tools list will be
	// cached and only fetched from the server once. If `false`, the tools list will be
	// fetched from the server on each call to `ListTools()`. The cache can be
	// invalidated by calling `InvalidateToolsCache()`. You should set this to `true`
	// if you know the server will not change its tools list, because it can drastically
	// improve latency (by avoiding a round-trip to the server every time).
	CacheToolsList bool

	// A readable name for the server. If not provided, we'll create one from the command.
	Name string

	// Optional tool filter to use for filtering tools
	ToolFilter MCPToolFilter

	// Whether to use `StructuredContent` when calling an MCP tool.
	// Defaults to false for backwards compatibility - most MCP servers still include
	// the structured content in `Content`, and using it by default will cause duplicate
	// content. You can set this to true if you know the server will not duplicate
	// the structured content in `Content`.
	UseStructuredContent bool

	// Maximum number of retry attempts for ListTools/CallTool. Use -1 for unlimited retries.
	MaxRetryAttempts int

	// Base delay for exponential backoff between retries.
	RetryBackoffBase time.Duration

	// Optional approval policy for MCP tools on this server.
	RequireApproval any

	// Optional resolver for MCP request metadata (`_meta`) on tool calls.
	ToolMetaResolver MCPToolMetaResolver

	// Optional per-server override for MCP tool failure handling.
	FailureErrorFunction    *ToolErrorFunction
	FailureErrorFunctionSet bool
}

// MCPServerStdio is an MCP server implementation that uses the stdio transport.
//
// See: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio
type MCPServerStdio struct {
	*MCPServerWithClientSession
}

// NewMCPServerStdio creates a new MCP server based on the stdio transport.
func NewMCPServerStdio(params MCPServerStdioParams) *MCPServerStdio {
	name := params.Name
	if name == "" {
		name = fmt.Sprintf("stdio: %s", params.Command.Path)
	}

	return &MCPServerStdio{
		MCPServerWithClientSession: NewMCPServerWithClientSession(MCPServerWithClientSessionParams{
			Name:                    name,
			Transport:               &mcp.CommandTransport{Command: params.Command},
			CacheToolsList:          params.CacheToolsList,
			ToolFilter:              params.ToolFilter,
			UseStructuredContent:    params.UseStructuredContent,
			ClientOptions:           params.ClientOptions,
			ClientSessionTimeout:    params.ClientSessionTimeout,
			MaxRetryAttempts:        params.MaxRetryAttempts,
			RetryBackoffBase:        params.RetryBackoffBase,
			RequireApproval:         params.RequireApproval,
			ToolMetaResolver:        params.ToolMetaResolver,
			FailureErrorFunction:    params.FailureErrorFunction,
			FailureErrorFunctionSet: params.FailureErrorFunctionSet,
		}),
	}
}

type MCPServerSSEParams struct {
	BaseURL       string
	TransportOpts *mcp.SSEClientTransport

	// Optional factory to build an HTTP client for SSE transport.
	HTTPClientFactory func() *http.Client

	// Optional static headers to attach to every SSE request.
	Headers map[string]string

	// Optional request authenticator applied before the request is sent.
	Auth MCPRequestAuthenticator

	// Optional request timeout for SSE transport.
	// Defaults to 5 seconds.
	Timeout time.Duration

	// Optional SSE idle read timeout.
	// Defaults to 5 minutes.
	SSEReadTimeout time.Duration

	// Optional client options, including MCP message handlers.
	ClientOptions *mcp.ClientOptions

	// Optional per-request timeout used for MCP client session calls.
	ClientSessionTimeout time.Duration

	// Whether to cache the tools list. If `true`, the tools list will be
	// cached and only fetched from the server once. If `false`, the tools list will be
	// fetched from the server on each call to `ListTools()`. The cache can be
	// invalidated by calling `InvalidateToolsCache()`. You should set this to `true`
	// if you know the server will not change its tools list, because it can drastically
	// improve latency (by avoiding a round-trip to the server every time).
	CacheToolsList bool

	// A readable name for the server. If not provided, we'll create one from the base URL.
	Name string

	// Optional tool filter to use for filtering tools
	ToolFilter MCPToolFilter

	// Whether to use `StructuredContent` when calling an MCP tool.
	// Defaults to false for backwards compatibility - most MCP servers still include
	// the structured content in `Content`, and using it by default will cause duplicate
	// content. You can set this to true if you know the server will not duplicate
	// the structured content in `Content`.
	UseStructuredContent bool

	// Maximum number of retry attempts for ListTools/CallTool. Use -1 for unlimited retries.
	MaxRetryAttempts int

	// Base delay for exponential backoff between retries.
	RetryBackoffBase time.Duration

	// Optional approval policy for MCP tools on this server.
	RequireApproval any

	// Optional resolver for MCP request metadata (`_meta`) on tool calls.
	ToolMetaResolver MCPToolMetaResolver

	// Optional per-server override for MCP tool failure handling.
	FailureErrorFunction    *ToolErrorFunction
	FailureErrorFunctionSet bool
}

// MCPServerSSE is an MCP server implementation that uses the HTTP with SSE transport.
//
// See: https://modelcontextprotocol.io/specification/2024-11-05/basic/transports#http-with-sse
//
// Deprecated: SSE as a standalone transport is deprecated as of MCP protocol version 2024-11-05.
// It has been replaced by Streamable HTTP, which incorporates SSE as an optional streaming mechanism.
// See: https://modelcontextprotocol.io/docs/concepts/transports#server-sent-events-sse-deprecated
type MCPServerSSE struct {
	*MCPServerWithClientSession
	headers        map[string]string
	timeout        time.Duration
	sseReadTimeout time.Duration
}

// NewMCPServerSSE creates a new MCP server based on the HTTP with SSE transport.
//
// Deprecated: SSE as a standalone transport is deprecated as of MCP protocol version 2024-11-05.
// It has been replaced by Streamable HTTP, which incorporates SSE as an optional streaming mechanism.
// See: https://modelcontextprotocol.io/docs/concepts/transports#server-sent-events-sse-deprecated
func NewMCPServerSSE(params MCPServerSSEParams) *MCPServerSSE {
	name := params.Name
	if name == "" {
		name = fmt.Sprintf("sse: %s", params.BaseURL)
	}
	timeout := params.Timeout
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	sseReadTimeout := params.SSEReadTimeout
	if sseReadTimeout <= 0 {
		sseReadTimeout = 300 * time.Second
	}

	transport := &mcp.SSEClientTransport{
		Endpoint: params.BaseURL,
	}
	if params.TransportOpts != nil {
		transport = &mcp.SSEClientTransport{
			Endpoint:   params.BaseURL,
			HTTPClient: params.TransportOpts.HTTPClient,
		}
	}
	if transport.HTTPClient == nil && params.HTTPClientFactory != nil {
		transport.HTTPClient = params.HTTPClientFactory()
	}
	transport.HTTPClient = decorateMCPHTTPClient(
		transport.HTTPClient,
		params.Headers,
		params.Auth,
		timeout,
		sseReadTimeout,
	)
	return &MCPServerSSE{
		MCPServerWithClientSession: NewMCPServerWithClientSession(MCPServerWithClientSessionParams{
			Name:                    name,
			Transport:               transport,
			CacheToolsList:          params.CacheToolsList,
			ToolFilter:              params.ToolFilter,
			UseStructuredContent:    params.UseStructuredContent,
			ClientOptions:           params.ClientOptions,
			ClientSessionTimeout:    params.ClientSessionTimeout,
			MaxRetryAttempts:        params.MaxRetryAttempts,
			RetryBackoffBase:        params.RetryBackoffBase,
			RequireApproval:         params.RequireApproval,
			ToolMetaResolver:        params.ToolMetaResolver,
			FailureErrorFunction:    params.FailureErrorFunction,
			FailureErrorFunctionSet: params.FailureErrorFunctionSet,
		}),
		headers:        maps.Clone(params.Headers),
		timeout:        timeout,
		sseReadTimeout: sseReadTimeout,
	}
}

type MCPServerStreamableHTTPParams struct {
	URL           string
	TransportOpts *mcp.StreamableClientTransport
	// Optional factory to build an HTTP client for Streamable HTTP transport.
	HTTPClientFactory func() *http.Client

	// Optional factory receiving resolved headers/timeout configuration.
	HTTPClientFactoryWithConfig func(headers map[string]string, timeout time.Duration) *http.Client

	// Optional static headers to attach to every Streamable HTTP request.
	Headers map[string]string

	// Optional request authenticator applied before the request is sent.
	Auth MCPRequestAuthenticator

	// Optional request timeout for Streamable HTTP transport.
	// Defaults to 5 seconds.
	Timeout time.Duration

	// Optional SSE idle read timeout. Implemented at the HTTP response body layer
	// because the Go MCP transport does not expose a direct field.
	SSEReadTimeout time.Duration

	// Optional terminate-on-close behavior for parity with Python configuration.
	// Currently stored for observability; Go MCP transport does not expose a direct field.
	TerminateOnClose *bool

	// Optional client options, including MCP message handlers.
	ClientOptions *mcp.ClientOptions

	// Optional per-request timeout used for MCP client session calls.
	ClientSessionTimeout time.Duration

	// Whether to cache the tools list. If `true`, the tools list will be
	// cached and only fetched from the server once. If `false`, the tools list will be
	// fetched from the server on each call to `ListTools()`. The cache can be
	// invalidated by calling `InvalidateToolsCache()`. You should set this to `true`
	// if you know the server will not change its tools list, because it can drastically
	// improve latency (by avoiding a round-trip to the server every time).
	CacheToolsList bool

	// A readable name for the server. If not provided, we'll create one from the URL.
	Name string

	// Optional tool filter to use for filtering tools
	ToolFilter MCPToolFilter

	// Whether to use `StructuredContent` when calling an MCP tool.
	// Defaults to false for backwards compatibility - most MCP servers still include
	// the structured content in `Content`, and using it by default will cause duplicate
	// content. You can set this to true if you know the server will not duplicate
	// the structured content in `Content`.
	UseStructuredContent bool

	// Maximum number of retry attempts for ListTools/CallTool. Use -1 for unlimited retries.
	MaxRetryAttempts int

	// Base delay for exponential backoff between retries.
	RetryBackoffBase time.Duration

	// Optional approval policy for MCP tools on this server.
	RequireApproval any

	// Optional resolver for MCP request metadata (`_meta`) on tool calls.
	ToolMetaResolver MCPToolMetaResolver

	// Optional per-server override for MCP tool failure handling.
	FailureErrorFunction    *ToolErrorFunction
	FailureErrorFunctionSet bool
}

// MCPServerStreamableHTTP is an MCP server implementation that uses the Streamable HTTP transport.
//
// See: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#streamable-http
type MCPServerStreamableHTTP struct {
	*MCPServerWithClientSession
	headers                map[string]string
	timeout                time.Duration
	sseReadTimeout         time.Duration
	terminateOnClose       bool
	transportFactory       func() mcp.Transport
	isolatedSessionFactory func(context.Context) (mcpClientSession, error)
}

func NewMCPServerStreamableHTTP(params MCPServerStreamableHTTPParams) *MCPServerStreamableHTTP {
	name := params.Name
	if name == "" {
		name = fmt.Sprintf("streamable_http: %s", params.URL)
	}

	timeout := params.Timeout
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	sseReadTimeout := params.SSEReadTimeout
	if sseReadTimeout <= 0 {
		sseReadTimeout = 300 * time.Second
	}
	terminateOnClose := true
	if params.TerminateOnClose != nil {
		terminateOnClose = *params.TerminateOnClose
	}

	buildTransport := func() mcp.Transport {
		transport := &mcp.StreamableClientTransport{
			Endpoint: params.URL,
		}
		if params.TransportOpts != nil {
			transport = &mcp.StreamableClientTransport{
				Endpoint:   params.URL,
				HTTPClient: params.TransportOpts.HTTPClient,
				MaxRetries: params.TransportOpts.MaxRetries,
			}
		}

		if transport.HTTPClient == nil {
			switch {
			case params.HTTPClientFactoryWithConfig != nil:
				transport.HTTPClient = params.HTTPClientFactoryWithConfig(maps.Clone(params.Headers), timeout)
			case params.HTTPClientFactory != nil:
				transport.HTTPClient = params.HTTPClientFactory()
			default:
				transport.HTTPClient = &http.Client{}
			}
		}

		headers := params.Headers
		if params.HTTPClientFactoryWithConfig != nil {
			headers = nil
		}
		transport.HTTPClient = decorateMCPHTTPClient(
			transport.HTTPClient,
			headers,
			params.Auth,
			timeout,
			sseReadTimeout,
		)
		return transport
	}
	transport := buildTransport()

	server := &MCPServerStreamableHTTP{
		MCPServerWithClientSession: NewMCPServerWithClientSession(MCPServerWithClientSessionParams{
			Name:                    name,
			Transport:               transport,
			CacheToolsList:          params.CacheToolsList,
			ToolFilter:              params.ToolFilter,
			UseStructuredContent:    params.UseStructuredContent,
			ClientOptions:           params.ClientOptions,
			ClientSessionTimeout:    params.ClientSessionTimeout,
			MaxRetryAttempts:        params.MaxRetryAttempts,
			RetryBackoffBase:        params.RetryBackoffBase,
			RequireApproval:         params.RequireApproval,
			ToolMetaResolver:        params.ToolMetaResolver,
			FailureErrorFunction:    params.FailureErrorFunction,
			FailureErrorFunctionSet: params.FailureErrorFunctionSet,
		}),
		headers:          maps.Clone(params.Headers),
		timeout:          timeout,
		sseReadTimeout:   sseReadTimeout,
		terminateOnClose: terminateOnClose,
		transportFactory: buildTransport,
	}
	server.requestSerializationSlot = newRequestSerializationSlot()
	return server
}

type sharedSessionRequestNeedsIsolation struct {
	err error
}

func (e *sharedSessionRequestNeedsIsolation) Error() string {
	if e == nil || e.err == nil {
		return ""
	}
	return e.err.Error()
}

func (e *sharedSessionRequestNeedsIsolation) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.err
}

type isolatedSessionRetryFailed struct {
	err error
}

func (e *isolatedSessionRetryFailed) Error() string {
	if e == nil || e.err == nil {
		return ""
	}
	return e.err.Error()
}

func (e *isolatedSessionRetryFailed) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.err
}

func (s *MCPServerStreamableHTTP) newIsolatedClientSession(ctx context.Context) (mcpClientSession, error) {
	if s.isolatedSessionFactory != nil {
		return s.isolatedSessionFactory(ctx)
	}
	if s.transportFactory == nil {
		return nil, NewUserError("streamable HTTP transport factory is not configured")
	}
	client := newMCPClient(&mcp.Implementation{Name: s.name}, s.clientOptions)
	session, err := client.Connect(ctx, s.transportFactory(), nil)
	if err != nil {
		return nil, fmt.Errorf("MCP client connection error: %w", err)
	}
	return session, nil
}

func (s *MCPServerStreamableHTTP) callToolWithSession(
	ctx context.Context,
	session mcpClientSession,
	toolName string,
	arguments map[string]any,
	meta map[string]any,
) (*mcp.CallToolResult, error) {
	attemptCtx, cancel := s.withSessionTimeout(ctx)
	defer cancel()

	params := &mcp.CallToolParams{
		Name:      toolName,
		Arguments: arguments,
	}
	if len(meta) > 0 {
		params.Meta = mcp.Meta(meta)
	}
	return session.CallTool(attemptCtx, params)
}

func (s *MCPServerStreamableHTTP) shouldRetryInIsolatedSession(err error) bool {
	if err == nil {
		return false
	}
	if multi, ok := err.(interface{ Unwrap() []error }); ok {
		errs := multi.Unwrap()
		return len(errs) > 0 && allErrors(errs, s.shouldRetryInIsolatedSession)
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, mcp.ErrConnectionClosed) {
		return true
	}
	statusCode := getStatusCodeFromError(err)
	if statusCode == 0 {
		statusCode = httpStatusCodeFromErrorMessage(err.Error())
	}
	if statusCode >= 500 || statusCode == http.StatusRequestTimeout {
		return true
	}
	if mcpCode, ok := intFromField(err, "Code"); ok && mcpCode == http.StatusRequestTimeout {
		return true
	}

	message := strings.ToLower(strings.TrimSpace(err.Error()))
	if strings.Contains(message, "timed out") ||
		strings.Contains(message, "timeout") ||
		strings.Contains(message, "deadline exceeded") ||
		strings.Contains(message, "connection closed") ||
		strings.Contains(message, "closed resource") {
		return true
	}
	return false
}

func (s *MCPServerStreamableHTTP) callToolWithSharedSession(
	ctx context.Context,
	toolName string,
	arguments map[string]any,
	meta map[string]any,
	allowIsolatedRetry bool,
) (*mcp.CallToolResult, error) {
	session := s.session
	if session == nil {
		return nil, NewUserError("server not initialized: make sure you call `Connect()` first")
	}
	result, err := withSerializedSessionRequest(
		ctx,
		s.requestSerializationSlot,
		func() (*mcp.CallToolResult, error) {
			return s.callToolWithSession(ctx, session, toolName, arguments, meta)
		},
	)
	if err == nil {
		return result, nil
	}
	if ctx != nil && ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if allowIsolatedRetry && s.shouldRetryInIsolatedSession(err) {
		return nil, &sharedSessionRequestNeedsIsolation{err: err}
	}
	return nil, err
}

func (s *MCPServerStreamableHTTP) callToolWithIsolatedRetry(
	ctx context.Context,
	toolName string,
	arguments map[string]any,
	meta map[string]any,
	allowIsolatedRetry bool,
) (*mcp.CallToolResult, bool, error) {
	result, err := s.callToolWithSharedSession(ctx, toolName, arguments, meta, allowIsolatedRetry)
	if err == nil {
		return result, false, nil
	}
	var needsIsolation *sharedSessionRequestNeedsIsolation
	if !errors.As(err, &needsIsolation) {
		return nil, false, err
	}

	session, isolatedErr := s.newIsolatedClientSession(ctx)
	if isolatedErr != nil {
		return nil, false, &isolatedSessionRetryFailed{err: isolatedErr}
	}
	defer session.Close()

	result, isolatedErr = s.callToolWithSession(ctx, session, toolName, arguments, meta)
	if isolatedErr != nil {
		return nil, false, &isolatedSessionRetryFailed{err: isolatedErr}
	}
	return result, true, nil
}

func (s *MCPServerStreamableHTTP) CallTool(
	ctx context.Context,
	toolName string,
	arguments map[string]any,
	meta map[string]any,
) (*mcp.CallToolResult, error) {
	if s.session == nil {
		return nil, NewUserError("server not initialized: make sure you call `Connect()` first")
	}
	if err := s.validateCallToolArguments(toolName, arguments); err != nil {
		return nil, err
	}

	retriesUsed := 0
	firstAttempt := true
	for {
		if !firstAttempt && s.maxRetryAttempts != -1 {
			retriesUsed++
		}
		allowIsolatedRetry := s.maxRetryAttempts == -1 || retriesUsed < s.maxRetryAttempts
		result, usedIsolatedRetry, err := s.callToolWithIsolatedRetry(
			ctx,
			toolName,
			arguments,
			meta,
			allowIsolatedRetry,
		)
		if err == nil {
			if usedIsolatedRetry && s.maxRetryAttempts != -1 {
				retriesUsed++
			}
			return result, nil
		}
		if ctx != nil && ctx.Err() != nil {
			return nil, ctx.Err()
		}

		var isolatedErr *isolatedSessionRetryFailed
		if errors.As(err, &isolatedErr) {
			retriesUsed++
			if s.maxRetryAttempts != -1 && retriesUsed >= s.maxRetryAttempts {
				if isolatedErr.err != nil {
					return nil, isolatedErr.err
				}
				return nil, err
			}
			if backoffErr := s.sleepRetryBackoff(ctx, retriesUsed-1); backoffErr != nil {
				return nil, backoffErr
			}
			firstAttempt = false
			continue
		}

		if s.maxRetryAttempts != -1 && retriesUsed >= s.maxRetryAttempts {
			return nil, err
		}
		if backoffErr := s.sleepRetryBackoff(ctx, retriesUsed); backoffErr != nil {
			return nil, backoffErr
		}
		firstAttempt = false
	}
}

func allErrors(errs []error, pred func(error) bool) bool {
	for _, err := range errs {
		if err == nil || !pred(err) {
			return false
		}
	}
	return true
}

func httpStatusCodeFromErrorMessage(message string) int {
	for _, field := range strings.Fields(message) {
		if len(field) < 3 {
			continue
		}
		start := 0
		for start < len(field) && (field[start] < '0' || field[start] > '9') {
			start++
		}
		end := start
		for end < len(field) && field[end] >= '0' && field[end] <= '9' {
			end++
		}
		if end-start != 3 {
			continue
		}
		statusCode, err := strconv.Atoi(field[start:end])
		if err == nil && statusCode >= 100 && statusCode <= 599 {
			return statusCode
		}
	}
	return 0
}

// Headers returns configured SSE headers.
func (s *MCPServerSSE) Headers() map[string]string {
	return maps.Clone(s.headers)
}

// Timeout returns configured SSE request timeout.
func (s *MCPServerSSE) Timeout() time.Duration {
	return s.timeout
}

// SSEReadTimeout returns configured SSE idle read timeout.
func (s *MCPServerSSE) SSEReadTimeout() time.Duration {
	return s.sseReadTimeout
}

// Headers returns configured Streamable HTTP headers.
func (s *MCPServerStreamableHTTP) Headers() map[string]string {
	return maps.Clone(s.headers)
}

// Timeout returns configured Streamable HTTP request timeout.
func (s *MCPServerStreamableHTTP) Timeout() time.Duration {
	return s.timeout
}

// SSEReadTimeout returns configured Streamable HTTP SSE read timeout.
func (s *MCPServerStreamableHTTP) SSEReadTimeout() time.Duration {
	return s.sseReadTimeout
}

// TerminateOnClose reports configured Streamable HTTP terminate-on-close behavior.
func (s *MCPServerStreamableHTTP) TerminateOnClose() bool {
	return s.terminateOnClose
}

type mcpHeaderTransport struct {
	next                 http.RoundTripper
	headers              map[string]string
	auth                 MCPRequestAuthenticator
	requestTimeout       time.Duration
	sseReadTimeout       time.Duration
	timeoutNeedsFallback bool
}

func decorateMCPHTTPClient(
	client *http.Client,
	headers map[string]string,
	auth MCPRequestAuthenticator,
	requestTimeout time.Duration,
	sseReadTimeout time.Duration,
) *http.Client {
	if len(headers) == 0 && auth == nil && requestTimeout <= 0 && sseReadTimeout <= 0 {
		return client
	}
	if client == nil {
		client = &http.Client{}
	}
	next, timeoutNeedsFallback := mcpTransportWithTimeout(client.Transport, requestTimeout)
	decorated := *client
	decorated.Transport = &mcpHeaderTransport{
		next:                 next,
		headers:              maps.Clone(headers),
		auth:                 auth,
		requestTimeout:       requestTimeout,
		sseReadTimeout:       sseReadTimeout,
		timeoutNeedsFallback: timeoutNeedsFallback,
	}
	return &decorated
}

func (t *mcpHeaderTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	transport := t.next
	if transport == nil {
		transport = http.DefaultTransport
	}
	clone := req.Clone(req.Context())
	clone.Header = req.Header.Clone()
	if t.auth != nil {
		if err := t.auth(clone); err != nil {
			return nil, err
		}
	}
	isSSERequest := isMCPEventStreamRequest(clone)
	var cancel context.CancelFunc
	if t.timeoutNeedsFallback && t.requestTimeout > 0 && !isSSERequest {
		timeoutCtx, timeoutCancel := context.WithTimeout(clone.Context(), t.requestTimeout)
		cancel = timeoutCancel
		clone = clone.WithContext(timeoutCtx)
		clone.Header = clone.Header.Clone()
	}
	for key, value := range t.headers {
		if strings.TrimSpace(key) == "" {
			continue
		}
		clone.Header.Set(key, value)
	}
	resp, err := transport.RoundTrip(clone)
	if err != nil {
		if cancel != nil {
			cancel()
		}
		return nil, err
	}
	if resp == nil || resp.Body == nil {
		if cancel != nil {
			cancel()
		}
		return resp, nil
	}
	if cancel != nil {
		resp.Body = &mcpCancelOnCloseBody{
			ReadCloser: resp.Body,
			cancel:     cancel,
		}
	}
	if t.sseReadTimeout > 0 && isSSERequest {
		resp.Body = &mcpReadTimeoutBody{
			body:    resp.Body,
			timeout: t.sseReadTimeout,
		}
	}
	return resp, nil
}

func mcpTransportWithTimeout(next http.RoundTripper, requestTimeout time.Duration) (http.RoundTripper, bool) {
	if requestTimeout <= 0 {
		return next, false
	}
	if next == nil {
		if transport, ok := http.DefaultTransport.(*http.Transport); ok {
			clone := transport.Clone()
			if clone.ResponseHeaderTimeout == 0 {
				clone.ResponseHeaderTimeout = requestTimeout
			}
			return clone, false
		}
		return nil, true
	}
	transport, ok := next.(*http.Transport)
	if !ok {
		return next, true
	}
	clone := transport.Clone()
	if clone.ResponseHeaderTimeout == 0 {
		clone.ResponseHeaderTimeout = requestTimeout
	}
	return clone, false
}

func isMCPEventStreamRequest(req *http.Request) bool {
	for _, accept := range req.Header.Values("Accept") {
		for _, mediaType := range strings.Split(accept, ",") {
			mediaType = strings.TrimSpace(mediaType)
			if cut := strings.Index(mediaType, ";"); cut >= 0 {
				mediaType = strings.TrimSpace(mediaType[:cut])
			}
			if strings.EqualFold(mediaType, "text/event-stream") {
				return true
			}
		}
	}
	return false
}

type mcpCancelOnCloseBody struct {
	io.ReadCloser
	cancel context.CancelFunc
	once   sync.Once
}

func (b *mcpCancelOnCloseBody) cancelOnce() {
	if b == nil || b.cancel == nil {
		return
	}
	b.once.Do(b.cancel)
}

func (b *mcpCancelOnCloseBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if err != nil {
		b.cancelOnce()
	}
	return n, err
}

func (b *mcpCancelOnCloseBody) Close() error {
	err := b.ReadCloser.Close()
	b.cancelOnce()
	return err
}

type mcpReadTimeoutBody struct {
	body    io.ReadCloser
	timeout time.Duration
}

func (b *mcpReadTimeoutBody) Read(p []byte) (int, error) {
	if b == nil || b.body == nil {
		return 0, io.EOF
	}
	if b.timeout <= 0 {
		return b.body.Read(p)
	}

	type result struct {
		n   int
		err error
	}

	resultCh := make(chan result, 1)
	go func() {
		n, err := b.body.Read(p)
		resultCh <- result{n: n, err: err}
	}()

	timer := time.NewTimer(b.timeout)
	defer timer.Stop()

	select {
	case outcome := <-resultCh:
		return outcome.n, outcome.err
	case <-timer.C:
		_ = b.body.Close()
		return 0, mcpReadTimeoutError{timeout: b.timeout}
	}
}

func (b *mcpReadTimeoutBody) Close() error {
	if b == nil || b.body == nil {
		return nil
	}
	return b.body.Close()
}

type mcpReadTimeoutError struct {
	timeout time.Duration
}

func (e mcpReadTimeoutError) Error() string {
	return fmt.Sprintf("MCP SSE read timed out after %s", e.timeout)
}

func (mcpReadTimeoutError) Timeout() bool { return true }

func (mcpReadTimeoutError) Temporary() bool { return true }
