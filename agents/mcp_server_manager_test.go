// Copyright 2026 The NLP Odyssey Authors
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
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type taskBoundServer struct {
	name       string
	connectGID uint64
	cleaned    bool

	connectErr error
	cleanupErr error
}

func (s *taskBoundServer) Connect(context.Context) error {
	s.connectGID = currentGoroutineID()
	if s.connectErr != nil {
		return s.connectErr
	}
	return nil
}

func (s *taskBoundServer) Cleanup(context.Context) error {
	if s.connectGID == 0 {
		return errors.New("server was not connected")
	}
	if currentGoroutineID() != s.connectGID {
		return fmt.Errorf("attempted to cleanup in a different goroutine")
	}
	s.cleaned = true
	return s.cleanupErr
}

func (s *taskBoundServer) Name() string { return s.name }
func (*taskBoundServer) UseStructuredContent() bool {
	return false
}
func (*taskBoundServer) ListTools(context.Context, *Agent) ([]*mcp.Tool, error) {
	return nil, nil
}
func (*taskBoundServer) CallTool(context.Context, string, map[string]any, map[string]any) (*mcp.CallToolResult, error) {
	return nil, nil
}
func (*taskBoundServer) ListPrompts(context.Context) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}
func (*taskBoundServer) GetPrompt(context.Context, string, map[string]string) (*mcp.GetPromptResult, error) {
	return &mcp.GetPromptResult{}, nil
}

type flakyServer struct {
	failuresRemaining int
	connectCalls      int
}

func (s *flakyServer) Connect(context.Context) error {
	s.connectCalls++
	if s.failuresRemaining > 0 {
		s.failuresRemaining--
		return errors.New("connect failed")
	}
	return nil
}

func (*flakyServer) Cleanup(context.Context) error { return nil }
func (*flakyServer) Name() string                  { return "flaky" }
func (*flakyServer) UseStructuredContent() bool    { return false }
func (*flakyServer) ListTools(context.Context, *Agent) ([]*mcp.Tool, error) {
	return nil, nil
}
func (*flakyServer) CallTool(context.Context, string, map[string]any, map[string]any) (*mcp.CallToolResult, error) {
	return nil, nil
}
func (*flakyServer) ListPrompts(context.Context) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}
func (*flakyServer) GetPrompt(context.Context, string, map[string]string) (*mcp.GetPromptResult, error) {
	return &mcp.GetPromptResult{}, nil
}

type cleanupAwareServer struct {
	connectCalls int
	cleanupCalls int
}

func (s *cleanupAwareServer) Connect(context.Context) error {
	if s.connectCalls > s.cleanupCalls {
		return errors.New("connect called without cleanup")
	}
	s.connectCalls++
	return nil
}

func (s *cleanupAwareServer) Cleanup(context.Context) error {
	s.cleanupCalls++
	return nil
}

func (*cleanupAwareServer) Name() string               { return "cleanup-aware" }
func (*cleanupAwareServer) UseStructuredContent() bool { return false }
func (*cleanupAwareServer) ListTools(context.Context, *Agent) ([]*mcp.Tool, error) {
	return nil, nil
}
func (*cleanupAwareServer) CallTool(context.Context, string, map[string]any, map[string]any) (*mcp.CallToolResult, error) {
	return nil, nil
}
func (*cleanupAwareServer) ListPrompts(context.Context) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}
func (*cleanupAwareServer) GetPrompt(context.Context, string, map[string]string) (*mcp.GetPromptResult, error) {
	return &mcp.GetPromptResult{}, nil
}

type cancelledServer struct{}

func (*cancelledServer) Connect(context.Context) error { return context.Canceled }
func (*cancelledServer) Cleanup(context.Context) error { return nil }
func (*cancelledServer) Name() string                  { return "cancelled" }
func (*cancelledServer) UseStructuredContent() bool    { return false }
func (*cancelledServer) ListTools(context.Context, *Agent) ([]*mcp.Tool, error) {
	return nil, nil
}
func (*cancelledServer) CallTool(context.Context, string, map[string]any, map[string]any) (*mcp.CallToolResult, error) {
	return nil, nil
}
func (*cancelledServer) ListPrompts(context.Context) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}
func (*cancelledServer) GetPrompt(context.Context, string, map[string]string) (*mcp.GetPromptResult, error) {
	return &mcp.GetPromptResult{}, nil
}

type barrierConnectServer struct {
	name    string
	ready   *sync.WaitGroup
	release <-chan struct{}
}

func (s *barrierConnectServer) Connect(context.Context) error {
	s.ready.Done()
	<-s.release
	return nil
}

func (*barrierConnectServer) Cleanup(context.Context) error { return nil }
func (s *barrierConnectServer) Name() string                { return s.name }
func (*barrierConnectServer) UseStructuredContent() bool    { return false }
func (*barrierConnectServer) ListTools(context.Context, *Agent) ([]*mcp.Tool, error) {
	return nil, nil
}
func (*barrierConnectServer) CallTool(context.Context, string, map[string]any, map[string]any) (*mcp.CallToolResult, error) {
	return nil, nil
}
func (*barrierConnectServer) ListPrompts(context.Context) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}
func (*barrierConnectServer) GetPrompt(context.Context, string, map[string]string) (*mcp.GetPromptResult, error) {
	return &mcp.GetPromptResult{}, nil
}

func TestMCPServerManagerConnectAndCleanupSameWorkerGoroutine(t *testing.T) {
	server := &taskBoundServer{name: "task-bound"}
	manager := NewMCPServerManager([]MCPServer{server}, MCPServerManagerParams{
		ConnectInParallel: true,
	})

	_, err := manager.ConnectAll(t.Context())
	require.NoError(t, err)
	assert.Equal(t, []MCPServer{server}, manager.ActiveServers())

	require.NoError(t, manager.CleanupAll(t.Context()))
	assert.True(t, server.cleaned)
	assert.Empty(t, manager.workers)
}

func TestMCPServerManagerCrossGoroutineCleanupFailsWithoutManager(t *testing.T) {
	server := &taskBoundServer{name: "task-bound"}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = server.Connect(context.Background())
	}()
	wg.Wait()

	err := server.Cleanup(t.Context())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "different goroutine")
}

func TestMCPServerManagerReconnectFailedOnly(t *testing.T) {
	server := &flakyServer{failuresRemaining: 1}
	manager := NewMCPServerManager([]MCPServer{server}, MCPServerManagerParams{})

	_, err := manager.ConnectAll(t.Context())
	require.NoError(t, err)
	assert.Empty(t, manager.ActiveServers())
	assert.Equal(t, []MCPServer{server}, manager.FailedServers())

	_, err = manager.Reconnect(t.Context(), true)
	require.NoError(t, err)
	assert.Equal(t, []MCPServer{server}, manager.ActiveServers())
	assert.Empty(t, manager.FailedServers())
	assert.Equal(t, 2, server.connectCalls)
}

func TestMCPServerManagerConnectAllIdempotent(t *testing.T) {
	server := &cleanupAwareServer{}
	manager := NewMCPServerManager([]MCPServer{server}, MCPServerManagerParams{})

	_, err := manager.ConnectAll(t.Context())
	require.NoError(t, err)
	assert.Equal(t, 1, server.connectCalls)

	_, err = manager.ConnectAll(t.Context())
	require.NoError(t, err)
	assert.Equal(t, 1, server.connectCalls)
}

func TestMCPServerManagerStrictConnectCleansUpConnectedServers(t *testing.T) {
	connected := &taskBoundServer{name: "connected"}
	failing := &flakyServer{failuresRemaining: 1}
	manager := NewMCPServerManager([]MCPServer{connected, failing}, MCPServerManagerParams{
		Strict: true,
	})

	_, err := manager.ConnectAll(t.Context())
	require.Error(t, err)
	assert.True(t, connected.cleaned)
	assert.Empty(t, manager.ActiveServers())
	assert.Equal(t, []MCPServer{failing}, manager.FailedServers())
}

func TestMCPServerManagerParallelCancelledSuppressedInStrictMode(t *testing.T) {
	server := &cancelledServer{}
	manager := NewMCPServerManager([]MCPServer{server}, MCPServerManagerParams{
		Strict:            true,
		ConnectInParallel: true,
	})

	_, err := manager.ConnectAll(t.Context())
	require.NoError(t, err)
	assert.Empty(t, manager.ActiveServers())
	assert.Equal(t, []MCPServer{server}, manager.FailedServers())
}

func TestMCPServerManagerParallelCancelledPropagatesWhenUnsuppressed(t *testing.T) {
	server := &cancelledServer{}
	manager := NewMCPServerManager([]MCPServer{server}, MCPServerManagerParams{
		ConnectInParallel:         true,
		SuppressCancelledError:    false,
		SuppressCancelledErrorSet: true,
	})

	_, err := manager.ConnectAll(t.Context())
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
}

func TestMCPServerManagerParallelConnectNoConcurrentMapWrite(t *testing.T) {
	const serverCount = 32

	release := make(chan struct{})
	var ready sync.WaitGroup
	ready.Add(serverCount)

	servers := make([]MCPServer, 0, serverCount)
	for i := range serverCount {
		servers = append(servers, &barrierConnectServer{
			name:    fmt.Sprintf("s-%d", i),
			ready:   &ready,
			release: release,
		})
	}

	manager := NewMCPServerManager(servers, MCPServerManagerParams{ConnectInParallel: true})

	done := make(chan error, 1)
	go func() {
		_, err := manager.ConnectAll(t.Context())
		done <- err
	}()

	ready.Wait()
	close(release)

	require.NoError(t, <-done)
	assert.Len(t, manager.ActiveServers(), serverCount)
}

func currentGoroutineID() uint64 {
	buf := make([]byte, 64)
	n := runtime.Stack(buf, false)
	header := strings.TrimPrefix(string(buf[:n]), "goroutine ")
	idField := strings.Fields(header)[0]
	id, _ := strconv.ParseUint(idField, 10, 64)
	return id
}
