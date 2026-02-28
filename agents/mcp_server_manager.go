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
	"log/slog"
	"slices"
	"sync"
	"time"
)

const (
	defaultMCPConnectTimeout = 10 * time.Second
	defaultMCPCleanupTimeout = 10 * time.Second
)

// MCPServerManagerParams configures MCP server manager lifecycle behavior.
type MCPServerManagerParams struct {
	ConnectTimeout time.Duration
	CleanupTimeout time.Duration

	DropFailedServers    bool
	DropFailedServersSet bool

	Strict bool

	SuppressCancelledError    bool
	SuppressCancelledErrorSet bool

	ConnectInParallel bool
}

// MCPServerManager manages lifecycle and status of a group of MCP servers.
type MCPServerManager struct {
	allServers    []MCPServer
	activeServers []MCPServer

	connectTimeout         time.Duration
	cleanupTimeout         time.Duration
	dropFailedServers      bool
	strict                 bool
	suppressCancelledError bool
	connectInParallel      bool

	workers          map[MCPServer]*mcpServerWorker
	failedServers    []MCPServer
	failedServerSet  map[MCPServer]struct{}
	connectedServers map[MCPServer]struct{}
	errorsByServer   map[MCPServer]error
}

// NewMCPServerManager creates a manager for the provided servers.
func NewMCPServerManager(servers []MCPServer, params MCPServerManagerParams) *MCPServerManager {
	connectTimeout := params.ConnectTimeout
	if connectTimeout <= 0 {
		connectTimeout = defaultMCPConnectTimeout
	}
	cleanupTimeout := params.CleanupTimeout
	if cleanupTimeout <= 0 {
		cleanupTimeout = defaultMCPCleanupTimeout
	}

	dropFailedServers := true
	if params.DropFailedServersSet {
		dropFailedServers = params.DropFailedServers
	}

	suppressCancelled := true
	if params.SuppressCancelledErrorSet {
		suppressCancelled = params.SuppressCancelledError
	}

	manager := &MCPServerManager{
		allServers:             slices.Clone(servers),
		activeServers:          slices.Clone(servers),
		connectTimeout:         connectTimeout,
		cleanupTimeout:         cleanupTimeout,
		dropFailedServers:      dropFailedServers,
		strict:                 params.Strict,
		suppressCancelledError: suppressCancelled,
		connectInParallel:      params.ConnectInParallel,
		workers:                make(map[MCPServer]*mcpServerWorker),
		failedServerSet:        make(map[MCPServer]struct{}),
		connectedServers:       make(map[MCPServer]struct{}),
		errorsByServer:         make(map[MCPServer]error),
	}
	return manager
}

// ActiveServers returns currently active servers.
func (m *MCPServerManager) ActiveServers() []MCPServer {
	return slices.Clone(m.activeServers)
}

// AllServers returns all managed servers.
func (m *MCPServerManager) AllServers() []MCPServer {
	return slices.Clone(m.allServers)
}

// FailedServers returns servers that most recently failed to connect/cleanup.
func (m *MCPServerManager) FailedServers() []MCPServer {
	return slices.Clone(m.failedServers)
}

// Errors returns a copy of server lifecycle errors.
func (m *MCPServerManager) Errors() map[MCPServer]error {
	out := make(map[MCPServer]error, len(m.errorsByServer))
	for server, err := range m.errorsByServer {
		out[server] = err
	}
	return out
}

// ConnectAll attempts to connect all not-yet-connected servers.
func (m *MCPServerManager) ConnectAll(ctx context.Context) ([]MCPServer, error) {
	previousConnected := make(map[MCPServer]struct{}, len(m.connectedServers))
	for server := range m.connectedServers {
		previousConnected[server] = struct{}{}
	}
	previousActive := slices.Clone(m.activeServers)

	m.failedServers = nil
	m.failedServerSet = make(map[MCPServer]struct{})
	m.errorsByServer = make(map[MCPServer]error)

	serversToConnect := m.serversToConnect(m.allServers)
	connectedThisRound := make([]MCPServer, 0, len(serversToConnect))

	var connectErr error
	if m.connectInParallel {
		connectErr = m.connectAllParallel(ctx, serversToConnect)
	} else {
		for _, server := range serversToConnect {
			err := m.attemptConnect(ctx, server, m.strict)
			if err != nil {
				connectErr = err
				break
			}
			if _, failed := m.failedServerSet[server]; !failed {
				connectedThisRound = append(connectedThisRound, server)
			}
		}
	}

	if connectErr != nil {
		if m.connectInParallel {
			_ = m.cleanupServers(ctx, serversToConnect)
		} else {
			serversToCleanup := m.uniqueServers(append(slices.Clone(connectedThisRound), m.failedServers...))
			_ = m.cleanupServers(ctx, serversToCleanup)
		}
		if m.dropFailedServers {
			m.activeServers = m.filterActiveServers(previousConnected)
		} else {
			m.activeServers = previousActive
		}
		return m.ActiveServers(), connectErr
	}

	m.refreshActiveServers()
	return m.ActiveServers(), nil
}

// Reconnect retries server connections.
// If failedOnly is true, only previously failed servers are retried.
func (m *MCPServerManager) Reconnect(ctx context.Context, failedOnly bool) ([]MCPServer, error) {
	var serversToRetry []MCPServer
	if failedOnly {
		serversToRetry = m.uniqueServers(m.failedServers)
	} else {
		if err := m.CleanupAll(ctx); err != nil {
			return nil, err
		}
		serversToRetry = slices.Clone(m.allServers)
		m.failedServers = nil
		m.failedServerSet = make(map[MCPServer]struct{})
		m.errorsByServer = make(map[MCPServer]error)
	}

	serversToRetry = m.serversToConnect(serversToRetry)
	var reconnectErr error
	if m.connectInParallel {
		reconnectErr = m.connectAllParallel(ctx, serversToRetry)
	} else {
		for _, server := range serversToRetry {
			if err := m.attemptConnect(ctx, server, m.strict); err != nil {
				reconnectErr = err
				break
			}
		}
	}
	m.refreshActiveServers()
	return m.ActiveServers(), reconnectErr
}

// CleanupAll cleans up all managed servers in reverse order.
func (m *MCPServerManager) CleanupAll(ctx context.Context) error {
	for i := len(m.allServers) - 1; i >= 0; i-- {
		server := m.allServers[i]
		err := m.cleanupServer(ctx, server)
		if err == nil {
			continue
		}
		if errors.Is(err, context.Canceled) && !m.suppressCancelledError {
			return err
		}
		Logger().Error("Failed to cleanup MCP server",
			slog.String("serverName", server.Name()),
			slog.String("error", err.Error()),
		)
		m.errorsByServer[server] = err
	}
	return nil
}

func (m *MCPServerManager) connectAllParallel(ctx context.Context, servers []MCPServer) error {
	if len(servers) == 0 {
		return nil
	}
	type result struct {
		err error
	}
	results := make(chan result, len(servers))
	var wg sync.WaitGroup
	wg.Add(len(servers))
	for _, server := range servers {
		server := server
		go func() {
			defer wg.Done()
			results <- result{err: m.attemptConnect(ctx, server, false)}
		}()
	}
	wg.Wait()
	close(results)

	var errs []error
	for item := range results {
		if item.err != nil {
			errs = append(errs, item.err)
		}
	}

	if !m.suppressCancelledError {
		for _, err := range errs {
			if errors.Is(err, context.Canceled) {
				return err
			}
		}
	}
	for _, err := range errs {
		if err != nil && !errors.Is(err, context.Canceled) {
			return err
		}
	}

	if m.strict && len(m.failedServers) > 0 {
		for _, server := range m.failedServers {
			err := m.errorsByServer[server]
			if err == nil {
				continue
			}
			if m.suppressCancelledError && errors.Is(err, context.Canceled) {
				continue
			}
			return err
		}
		if !m.suppressCancelledError {
			if err := m.errorsByServer[m.failedServers[0]]; err != nil {
				return err
			}
		}
	}
	return nil
}

func (m *MCPServerManager) attemptConnect(ctx context.Context, server MCPServer, raiseOnError bool) error {
	err := m.runConnect(ctx, server)
	if err == nil {
		m.connectedServers[server] = struct{}{}
		if _, failed := m.failedServerSet[server]; failed {
			m.removeFailedServer(server)
			delete(m.errorsByServer, server)
		}
		return nil
	}

	if errors.Is(err, context.Canceled) {
		if !m.suppressCancelledError {
			return err
		}
		m.recordFailure(server, err, "connect")
		return nil
	}

	m.recordFailure(server, err, "connect")
	if raiseOnError {
		return err
	}
	return nil
}

func (m *MCPServerManager) runConnect(ctx context.Context, server MCPServer) error {
	if m.connectInParallel {
		worker := m.getWorker(server)
		return worker.connect(ctx)
	}
	return runServerActionWithTimeout(ctx, m.connectTimeout, server.Connect)
}

func (m *MCPServerManager) cleanupServers(ctx context.Context, servers []MCPServer) error {
	for i := len(servers) - 1; i >= 0; i-- {
		server := servers[i]
		err := m.cleanupServer(ctx, server)
		if err == nil {
			continue
		}
		if errors.Is(err, context.Canceled) && !m.suppressCancelledError {
			return err
		}
		m.errorsByServer[server] = err
	}
	return nil
}

func (m *MCPServerManager) cleanupServer(ctx context.Context, server MCPServer) error {
	if m.connectInParallel {
		worker, ok := m.workers[server]
		if ok {
			if worker.isDone() {
				delete(m.workers, server)
				delete(m.connectedServers, server)
				return nil
			}
			err := worker.cleanup(ctx)
			delete(m.workers, server)
			delete(m.connectedServers, server)
			return err
		}
	}
	defer delete(m.connectedServers, server)
	return runServerActionWithTimeout(ctx, m.cleanupTimeout, server.Cleanup)
}

func (m *MCPServerManager) recordFailure(server MCPServer, err error, phase string) {
	Logger().Error("MCP server lifecycle failure",
		slog.String("phase", phase),
		slog.String("serverName", server.Name()),
		slog.String("error", err.Error()),
	)
	if _, ok := m.failedServerSet[server]; !ok {
		m.failedServerSet[server] = struct{}{}
		m.failedServers = append(m.failedServers, server)
	}
	m.errorsByServer[server] = err
}

func (m *MCPServerManager) refreshActiveServers() {
	if !m.dropFailedServers {
		m.activeServers = slices.Clone(m.allServers)
		return
	}
	m.activeServers = m.filterActiveServers(nil)
}

func (m *MCPServerManager) filterActiveServers(allowed map[MCPServer]struct{}) []MCPServer {
	active := make([]MCPServer, 0, len(m.allServers))
	for _, server := range m.allServers {
		if allowed != nil {
			if _, ok := allowed[server]; !ok {
				continue
			}
		} else {
			if _, failed := m.failedServerSet[server]; failed {
				continue
			}
		}
		active = append(active, server)
	}
	return active
}

func (m *MCPServerManager) getWorker(server MCPServer) *mcpServerWorker {
	if worker, ok := m.workers[server]; ok && !worker.isDone() {
		return worker
	}
	worker := newMCPServerWorker(server, m.connectTimeout, m.cleanupTimeout)
	m.workers[server] = worker
	return worker
}

func (m *MCPServerManager) removeFailedServer(server MCPServer) {
	delete(m.failedServerSet, server)
	filtered := m.failedServers[:0]
	for _, failed := range m.failedServers {
		if failed == server {
			continue
		}
		filtered = append(filtered, failed)
	}
	m.failedServers = filtered
}

func (m *MCPServerManager) serversToConnect(servers []MCPServer) []MCPServer {
	unique := m.uniqueServers(servers)
	if len(m.connectedServers) == 0 {
		return unique
	}
	filtered := make([]MCPServer, 0, len(unique))
	for _, server := range unique {
		if _, ok := m.connectedServers[server]; ok {
			continue
		}
		filtered = append(filtered, server)
	}
	return filtered
}

func (m *MCPServerManager) uniqueServers(servers []MCPServer) []MCPServer {
	seen := make(map[MCPServer]struct{}, len(servers))
	unique := make([]MCPServer, 0, len(servers))
	for _, server := range servers {
		if _, ok := seen[server]; ok {
			continue
		}
		seen[server] = struct{}{}
		unique = append(unique, server)
	}
	return unique
}

type mcpServerCommand struct {
	action string
	ctx    context.Context
	result chan error
}

type mcpServerWorker struct {
	server         MCPServer
	connectTimeout time.Duration
	cleanupTimeout time.Duration
	commands       chan mcpServerCommand
	done           chan struct{}
}

func newMCPServerWorker(server MCPServer, connectTimeout, cleanupTimeout time.Duration) *mcpServerWorker {
	worker := &mcpServerWorker{
		server:         server,
		connectTimeout: connectTimeout,
		cleanupTimeout: cleanupTimeout,
		commands:       make(chan mcpServerCommand),
		done:           make(chan struct{}),
	}
	go worker.run()
	return worker
}

func (w *mcpServerWorker) connect(ctx context.Context) error {
	return w.submit("connect", ctx)
}

func (w *mcpServerWorker) cleanup(ctx context.Context) error {
	return w.submit("cleanup", ctx)
}

func (w *mcpServerWorker) submit(action string, ctx context.Context) error {
	result := make(chan error, 1)
	cmd := mcpServerCommand{action: action, ctx: ctx, result: result}
	select {
	case w.commands <- cmd:
	case <-w.done:
		return fmt.Errorf("worker closed")
	}
	return <-result
}

func (w *mcpServerWorker) isDone() bool {
	select {
	case <-w.done:
		return true
	default:
		return false
	}
}

func (w *mcpServerWorker) run() {
	defer close(w.done)
	for cmd := range w.commands {
		var err error
		switch cmd.action {
		case "connect":
			err = runServerActionWithTimeout(cmd.ctx, w.connectTimeout, w.server.Connect)
		case "cleanup":
			err = runServerActionWithTimeout(cmd.ctx, w.cleanupTimeout, w.server.Cleanup)
		default:
			err = fmt.Errorf("unknown command: %s", cmd.action)
		}
		cmd.result <- err
		if cmd.action == "cleanup" {
			return
		}
	}
}

func runServerActionWithTimeout(
	ctx context.Context,
	timeout time.Duration,
	fn func(context.Context) error,
) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if timeout <= 0 {
		return fn(ctx)
	}
	callCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return fn(callCtx)
}
