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
	"log/slog"
	"maps"
	"net/http"
	"os/exec"
	"slices"
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

// MCPServerWithClientSession is a base type for MCP servers that uses an
// mcp.ClientSession to communicate with the server.
type MCPServerWithClientSession struct {
	transport            mcp.Transport
	session              mcpClientSession
	cleanupMu            sync.Mutex
	cacheToolsList       bool
	cacheDirty           bool
	toolsList            []*mcp.Tool
	toolFilter           MCPToolFilter
	name                 string
	useStructuredContent bool
	maxRetryAttempts     int
	retryBackoffBase     time.Duration
	clientOptions        *mcp.ClientOptions
	clientSessionTimeout time.Duration
	needsApprovalPolicy  mcpNeedsApprovalPolicy
	failureErrorFunction *ToolErrorFunction
	failureErrorSet      bool
	toolMetaResolver     MCPToolMetaResolver
	cleanupHook          func()
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
			attemptCtx, cancel := s.withSessionTimeout(ctx)
			defer cancel()
			listToolsResults, err = s.session.ListTools(attemptCtx, nil)
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
		attemptCtx, cancel := s.withSessionTimeout(ctx)
		defer cancel()

		params := &mcp.CallToolParams{
			Name:      toolName,
			Arguments: arguments,
		}
		if len(meta) > 0 {
			params.Meta = mcp.Meta(meta)
		}
		result, err = s.session.CallTool(attemptCtx, params)
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
		if s.retryBackoffBase <= 0 {
			continue
		}
		backoff := s.retryBackoffBase * time.Duration(1<<maxInt(attempts-1, 0))
		timer := time.NewTimer(backoff)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
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
	attemptCtx, cancel := s.withSessionTimeout(ctx)
	defer cancel()
	return s.session.ListPrompts(attemptCtx, nil)
}

func (s *MCPServerWithClientSession) GetPrompt(ctx context.Context, name string, arguments map[string]string) (*mcp.GetPromptResult, error) {
	if s.session == nil {
		return nil, NewUserError("server not initialized: make sure you call `Connect()` first")
	}
	attemptCtx, cancel := s.withSessionTimeout(ctx)
	defer cancel()
	return s.session.GetPrompt(attemptCtx, &mcp.GetPromptParams{
		Name:      name,
		Arguments: arguments,
	})
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

	transport := &mcp.SSEClientTransport{
		Endpoint: params.BaseURL,
	}
	if params.TransportOpts != nil {
		transport = &mcp.SSEClientTransport{
			Endpoint:   params.BaseURL,
			HTTPClient: params.TransportOpts.HTTPClient,
		}
	}
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

	// Optional request timeout for Streamable HTTP transport.
	// Defaults to 5 seconds.
	Timeout time.Duration

	// Optional SSE read timeout setting for parity with Python configuration.
	// Currently stored for observability; Go MCP transport does not expose a direct field.
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
	headers          map[string]string
	timeout          time.Duration
	sseReadTimeout   time.Duration
	terminateOnClose bool
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
			transport.HTTPClient = &http.Client{Timeout: timeout}
		}
	} else if timeout > 0 && transport.HTTPClient.Timeout == 0 {
		transport.HTTPClient.Timeout = timeout
	}

	if len(params.Headers) > 0 {
		transport.HTTPClient.Transport = &mcpHeaderTransport{
			next:    transport.HTTPClient.Transport,
			headers: maps.Clone(params.Headers),
		}
	}

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
	}
	return server
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
	next    http.RoundTripper
	headers map[string]string
}

func (t *mcpHeaderTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	transport := t.next
	if transport == nil {
		transport = http.DefaultTransport
	}
	clone := req.Clone(req.Context())
	clone.Header = req.Header.Clone()
	for key, value := range t.headers {
		if strings.TrimSpace(key) == "" {
			continue
		}
		clone.Header.Set(key, value)
	}
	return transport.RoundTrip(clone)
}
