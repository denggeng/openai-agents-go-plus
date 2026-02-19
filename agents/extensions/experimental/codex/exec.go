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

package codex

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/denggeng/openai-agents-go-plus/agents"
)

const (
	internalOriginatorEnv        = "CODEX_INTERNAL_ORIGINATOR_OVERRIDE"
	typescriptSDKOriginator      = "codex_sdk_ts"
	subprocessStreamLimitEnvVar  = "OPENAI_AGENTS_CODEX_SUBPROCESS_STREAM_LIMIT_BYTES"
	defaultSubprocessStreamLimit = 8 * 1024 * 1024
	minSubprocessStreamLimit     = 64 * 1024
	maxSubprocessStreamLimit     = 64 * 1024 * 1024
)

// CodexExec stores resolved executable and process environment settings.
type CodexExec struct {
	executablePath             string
	envOverride                map[string]string
	subprocessStreamLimitBytes int
}

// CodexExecArgs describes one Codex CLI execution request.
type CodexExecArgs struct {
	Input                 string
	BaseURL               *string
	APIKey                *string
	ThreadID              *string
	Images                []string
	Model                 *string
	SandboxMode           *string
	WorkingDirectory      *string
	AdditionalDirectories []string
	SkipGitRepoCheck      *bool
	OutputSchemaFile      *string
	ModelReasoningEffort  *string
	Signal                <-chan struct{}
	IdleTimeoutSeconds    *float64
	NetworkAccessEnabled  *bool
	WebSearchMode         *string
	WebSearchEnabled      *bool
	ApprovalPolicy        *string
}

// CodexExecClient is the execution abstraction used by Thread.
type CodexExecClient interface {
	RunJSONL(ctx context.Context, args CodexExecArgs) (<-chan string, <-chan error)
}

// NewCodexExec builds Codex execution settings from constructor options and environment.
func NewCodexExec(
	executablePath *string,
	env map[string]string,
	subprocessStreamLimitBytes *int,
) (*CodexExec, error) {
	resolvedLimit, err := resolveSubprocessStreamLimitBytes(subprocessStreamLimitBytes)
	if err != nil {
		return nil, err
	}
	return &CodexExec{
		executablePath:             resolveCodexPath(executablePath),
		envOverride:                cloneStringStringMap(env),
		subprocessStreamLimitBytes: resolvedLimit,
	}, nil
}

func resolveCodexPath(executablePath *string) string {
	if executablePath != nil && strings.TrimSpace(*executablePath) != "" {
		return *executablePath
	}
	return findCodexPath()
}

var (
	lookPath     = exec.LookPath
	runtimeOS    = func() string { return runtime.GOOS }
	runtimeArch  = func() string { return runtime.GOARCH }
	execFilePath = func() string {
		_, file, _, ok := runtime.Caller(0)
		if !ok {
			return ""
		}
		return file
	}
)

func findCodexPath() string {
	if fromEnv := strings.TrimSpace(os.Getenv("CODEX_PATH")); fromEnv != "" {
		return fromEnv
	}
	if lookedUp, err := lookPath("codex"); err == nil && strings.TrimSpace(lookedUp) != "" {
		return lookedUp
	}
	triple, err := platformTargetTriple(runtimeOS(), runtimeArch())
	if err != nil {
		return "codex"
	}
	root := codexVendorRoot()
	if root == "" {
		return "codex"
	}
	return filepath.Join(root, "vendor", triple, "codex", "codex")
}

func codexVendorRoot() string {
	file := execFilePath()
	if strings.TrimSpace(file) == "" {
		return ""
	}
	dir := filepath.Dir(file)
	for i := 0; i < 2; i++ {
		dir = filepath.Dir(dir)
	}
	if strings.TrimSpace(dir) == "" || dir == "." {
		return ""
	}
	return dir
}

func platformTargetTriple(system, arch string) (string, error) {
	switch system {
	case "linux":
		switch arch {
		case "amd64", "x86_64":
			return "x86_64-unknown-linux-musl", nil
		case "arm64", "aarch64":
			return "aarch64-unknown-linux-musl", nil
		}
	case "darwin":
		switch arch {
		case "amd64", "x86_64":
			return "x86_64-apple-darwin", nil
		case "arm64", "aarch64":
			return "aarch64-apple-darwin", nil
		}
	case "windows", "win32":
		switch arch {
		case "amd64", "x86_64":
			return "x86_64-pc-windows-msvc", nil
		case "arm64", "aarch64":
			return "aarch64-pc-windows-msvc", nil
		}
	}
	return "", fmt.Errorf("Unsupported platform: %s/%s", system, arch)
}

func resolveSubprocessStreamLimitBytes(explicitValue *int) (int, error) {
	if explicitValue != nil {
		return validateSubprocessStreamLimitBytes(*explicitValue)
	}

	fromEnv := strings.TrimSpace(os.Getenv(subprocessStreamLimitEnvVar))
	if fromEnv == "" {
		return defaultSubprocessStreamLimit, nil
	}

	parsed, err := strconv.Atoi(fromEnv)
	if err != nil {
		return 0, agents.UserErrorf("%s must be an integer number of bytes.", subprocessStreamLimitEnvVar)
	}
	return validateSubprocessStreamLimitBytes(parsed)
}

func validateSubprocessStreamLimitBytes(value int) (int, error) {
	if value < minSubprocessStreamLimit || value > maxSubprocessStreamLimit {
		return 0, agents.UserErrorf(
			"codex_subprocess_stream_limit_bytes must be between %d and %d bytes.",
			minSubprocessStreamLimit,
			maxSubprocessStreamLimit,
		)
	}
	return value, nil
}

func cloneStringStringMap(value map[string]string) map[string]string {
	if value == nil {
		return nil
	}
	out := make(map[string]string, len(value))
	for key, each := range value {
		out[key] = each
	}
	return out
}

// RunJSONL streams JSONL events from the Codex CLI.
func (c *CodexExec) RunJSONL(ctx context.Context, args CodexExecArgs) (<-chan string, <-chan error) {
	lines := make(chan string)
	errs := make(chan error, 1)

	go func() {
		defer close(lines)
		defer close(errs)

		commandArgs := c.buildCommandArgs(args)
		cmd := exec.CommandContext(ctx, c.executablePath, commandArgs...)
		cmd.Env = mapToEnv(c.buildEnv(args))

		stdin, err := cmd.StdinPipe()
		if err != nil {
			trySendError(errs, err)
			return
		}
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			trySendError(errs, err)
			return
		}
		stderr, err := cmd.StderrPipe()
		if err != nil {
			trySendError(errs, err)
			return
		}

		if err := cmd.Start(); err != nil {
			trySendError(errs, err)
			return
		}

		var stderrBuffer bytes.Buffer
		stderrDone := make(chan struct{})
		go func() {
			defer close(stderrDone)
			_, _ = io.Copy(&stderrBuffer, stderr)
		}()

		if _, err := io.WriteString(stdin, args.Input); err != nil {
			_ = stdin.Close()
			_ = killProcess(cmd)
			_ = cmd.Wait()
			<-stderrDone
			trySendError(errs, err)
			return
		}
		if err := stdin.Close(); err != nil {
			_ = killProcess(cmd)
			_ = cmd.Wait()
			<-stderrDone
			trySendError(errs, err)
			return
		}

		stopSignalWatch := startSignalWatcher(args.Signal, cmd)
		defer stopSignalWatch()

		readStop := make(chan struct{})
		scannedLines := make(chan string, 16)
		scanErrs := make(chan error, 1)
		go scanJSONLLines(stdout, c.subprocessStreamLimitBytes, scannedLines, scanErrs, readStop)

		idleDuration := durationFromIdleTimeout(args.IdleTimeoutSeconds)
		var idleTimer *time.Timer
		var idleTimeout <-chan time.Time
		if idleDuration > 0 {
			idleTimer = time.NewTimer(idleDuration)
			idleTimeout = idleTimer.C
		}
		stopIdleTimer := func() {
			if idleTimer == nil {
				return
			}
			if !idleTimer.Stop() {
				select {
				case <-idleTimer.C:
				default:
				}
			}
		}
		resetIdleTimer := func() {
			if idleTimer == nil {
				return
			}
			stopIdleTimer()
			idleTimer.Reset(idleDuration)
		}
		defer stopIdleTimer()

		var scanErr error
		linesCh := scannedLines
		scanErrCh := scanErrs
		for linesCh != nil || scanErrCh != nil {
			select {
			case <-ctx.Done():
				close(readStop)
				_ = killProcess(cmd)
				_ = cmd.Wait()
				<-stderrDone
				trySendError(errs, ctx.Err())
				return
			case <-idleTimeout:
				close(readStop)
				_ = killProcess(cmd)
				_ = cmd.Wait()
				<-stderrDone
				trySendError(errs, idleTimeoutError(args.IdleTimeoutSeconds))
				return
			case line, ok := <-linesCh:
				if !ok {
					linesCh = nil
					continue
				}
				resetIdleTimer()
				select {
				case lines <- line:
				case <-ctx.Done():
					close(readStop)
					_ = killProcess(cmd)
					_ = cmd.Wait()
					<-stderrDone
					trySendError(errs, ctx.Err())
					return
				}
			case err, ok := <-scanErrCh:
				if !ok {
					scanErrCh = nil
					continue
				}
				resetIdleTimer()
				if err != nil {
					scanErr = err
				}
			}
		}
		close(readStop)

		waitErr := cmd.Wait()
		<-stderrDone
		if scanErr != nil {
			trySendError(errs, scanErr)
			return
		}
		if waitErr != nil {
			exitCode := exitCodeFromErr(waitErr)
			stderrText := stderrBuffer.String()
			if exitCode != nil {
				trySendError(errs, fmt.Errorf("Codex exec exited with code %d: %s", *exitCode, stderrText))
				return
			}
			trySendError(errs, fmt.Errorf("Codex exec failed: %w: %s", waitErr, stderrText))
		}
	}()

	return lines, errs
}

func (c *CodexExec) buildCommandArgs(args CodexExecArgs) []string {
	commandArgs := []string{"exec", "--experimental-json"}
	if value := optionalValue(args.Model); value != "" {
		commandArgs = append(commandArgs, "--model", value)
	}
	if value := optionalValue(args.SandboxMode); value != "" {
		commandArgs = append(commandArgs, "--sandbox", value)
	}
	if value := optionalValue(args.WorkingDirectory); value != "" {
		commandArgs = append(commandArgs, "--cd", value)
	}
	for _, directory := range args.AdditionalDirectories {
		commandArgs = append(commandArgs, "--add-dir", directory)
	}
	if args.SkipGitRepoCheck != nil && *args.SkipGitRepoCheck {
		commandArgs = append(commandArgs, "--skip-git-repo-check")
	}
	if value := optionalValue(args.OutputSchemaFile); value != "" {
		commandArgs = append(commandArgs, "--output-schema", value)
	}
	if value := optionalValue(args.ModelReasoningEffort); value != "" {
		commandArgs = append(commandArgs, "--config", fmt.Sprintf("model_reasoning_effort=%q", value))
	}
	if args.NetworkAccessEnabled != nil {
		commandArgs = append(
			commandArgs,
			"--config",
			fmt.Sprintf("sandbox_workspace_write.network_access=%t", *args.NetworkAccessEnabled),
		)
	}
	if value := optionalValue(args.WebSearchMode); value != "" {
		commandArgs = append(commandArgs, "--config", fmt.Sprintf("web_search=%q", value))
	} else if args.WebSearchEnabled != nil {
		if *args.WebSearchEnabled {
			commandArgs = append(commandArgs, "--config", `web_search="live"`)
		} else {
			commandArgs = append(commandArgs, "--config", `web_search="disabled"`)
		}
	}
	if value := optionalValue(args.ApprovalPolicy); value != "" {
		commandArgs = append(commandArgs, "--config", fmt.Sprintf("approval_policy=%q", value))
	}
	if value := optionalValue(args.ThreadID); value != "" {
		commandArgs = append(commandArgs, "resume", value)
	}
	for _, image := range args.Images {
		commandArgs = append(commandArgs, "--image", image)
	}
	commandArgs = append(commandArgs, "-")
	return commandArgs
}

func (c *CodexExec) buildEnv(args CodexExecArgs) map[string]string {
	env := make(map[string]string)
	if c.envOverride != nil {
		for key, value := range c.envOverride {
			env[key] = value
		}
	} else {
		for _, each := range os.Environ() {
			key, value, ok := strings.Cut(each, "=")
			if !ok {
				continue
			}
			env[key] = value
		}
	}
	if _, ok := env[internalOriginatorEnv]; !ok {
		env[internalOriginatorEnv] = typescriptSDKOriginator
	}
	if value := optionalValue(args.BaseURL); value != "" {
		env["OPENAI_BASE_URL"] = value
	}
	if value := optionalValue(args.APIKey); value != "" {
		env["CODEX_API_KEY"] = value
	}
	return env
}

func mapToEnv(env map[string]string) []string {
	if len(env) == 0 {
		return nil
	}
	keys := make([]string, 0, len(env))
	for key := range env {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, key := range keys {
		out = append(out, key+"="+env[key])
	}
	return out
}

func optionalValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func scanJSONLLines(
	reader io.Reader,
	limit int,
	out chan<- string,
	errs chan<- error,
	stop <-chan struct{},
) {
	defer close(out)
	defer close(errs)

	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), limit)
	for scanner.Scan() {
		line := scanner.Text()
		select {
		case out <- line:
		case <-stop:
			return
		}
	}
	if err := scanner.Err(); err != nil {
		select {
		case errs <- err:
		case <-stop:
		}
	}
}

func startSignalWatcher(signal <-chan struct{}, cmd *exec.Cmd) func() {
	if signal == nil {
		return func() {}
	}
	stop := make(chan struct{})
	go func() {
		select {
		case <-signal:
			_ = killProcess(cmd)
		case <-stop:
		}
	}()
	return func() {
		close(stop)
	}
}

func killProcess(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	return cmd.Process.Kill()
}

func exitCodeFromErr(err error) *int {
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr == nil {
		return nil
	}
	code := exitErr.ExitCode()
	return &code
}
