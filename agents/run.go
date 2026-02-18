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
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"reflect"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/denggeng/openai-agents-go-plus/memory"
	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/denggeng/openai-agents-go-plus/tracing"
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

const DefaultMaxTurns = 10

// ModelInputData is a container for the data that will be sent to the model.
type ModelInputData struct {
	Input        []TResponseInputItem
	Instructions param.Opt[string]
}

// CallModelData contains data passed to RunConfig.CallModelInputFilter prior to model call.
type CallModelData struct {
	ModelData ModelInputData
	Agent     *Agent
}

// CallModelInputFilter is a type alias for the optional input filter callback.
type CallModelInputFilter = func(context.Context, CallModelData) (*ModelInputData, error)

// DefaultRunner is the default Runner instance used by package-level Run
// helpers.
var DefaultRunner = Runner{}

// Runner executes agents using the configured RunConfig.
//
// The zero value is valid.
type Runner struct {
	Config RunConfig
}

const DefaultWorkflowName = "Agent workflow"

// RunConfig configures settings for the entire agent run.
type RunConfig struct {
	// The model to use for the entire agent run. If set, will override the model set on every
	// agent. The ModelProvider passed in below must be able to resolve this model name.
	Model param.Opt[AgentModel]

	// Optional model provider to use when looking up string model names. Defaults to OpenAI (MultiProvider).
	ModelProvider ModelProvider

	// Optional global model settings. Any non-null or non-zero values will
	// override the agent-specific model settings.
	ModelSettings modelsettings.ModelSettings

	// Optional global input filter to apply to all handoffs. If `Handoff.InputFilter` is set, then that
	// will take precedence. The input filter allows you to edit the inputs that are sent to the new
	// agent. See the documentation in `Handoff.InputFilter` for more details.
	HandoffInputFilter HandoffInputFilter

	// A list of input guardrails to run on the initial run input.
	InputGuardrails []InputGuardrail

	// A list of output guardrails to run on the final output of the run.
	OutputGuardrails []OutputGuardrail

	// Optional callback that formats tool error messages, such as approval rejections.
	ToolErrorFormatter ToolErrorFormatter

	// Whether tracing is disabled for the agent run. If disabled, we will not trace the agent run.
	// Default: false (tracing enabled).
	TracingDisabled bool

	// Whether we include potentially sensitive data (for example: inputs/outputs of tool calls or
	// LLM generations) in traces. If false, we'll still create spans for these events, but the
	// sensitive data will not be included.
	// Default: true.
	TraceIncludeSensitiveData param.Opt[bool]

	// The name of the run, used for tracing. Should be a logical name for the run, like
	// "Code generation workflow" or "Customer support agent".
	// Default: DefaultWorkflowName.
	WorkflowName string

	// Optional custom trace ID to use for tracing.
	// If not provided, we will generate a new trace ID.
	TraceID string

	// Optional grouping identifier to use for tracing, to link multiple traces from the same conversation
	// or process. For example, you might use a chat thread ID.
	GroupID string

	// An optional dictionary of additional metadata to include with the trace.
	TraceMetadata map[string]any

	// Optional callback that is invoked immediately before calling the model. It receives the current
	// agent and the model input (instructions and input items), and must return a possibly
	// modified `ModelInputData` to use for the model call.
	//
	// This allows you to edit the input sent to the model e.g. to stay within a token limit.
	// For example, you can use this to add a system prompt to the input.
	CallModelInputFilter CallModelInputFilter

	// Optional maximum number of turns to run the agent for.
	// A turn is defined as one AI invocation (including any tool calls that might occur).
	// Default (when left zero): DefaultMaxTurns.
	MaxTurns uint64

	// Optional object that receives callbacks on various lifecycle events.
	Hooks RunHooks

	// Optional ID of the previous response, if using OpenAI models via the Responses API,
	// this allows you to skip passing in input from the previous turn.
	PreviousResponseID string

	// Optional conversation ID for server-managed conversation state.
	ConversationID string

	// Enable automatic previous response tracking for the first turn.
	AutoPreviousResponseID bool

	// Optional session for the run.
	Session memory.Session

	// Optional limit for the recover of the session of memory.
	LimitMemory int
}

// EventSeqResult contains the sequence of streaming events generated by
// RunStreamedSeq and the error, if any, that occurred while streaming.
type EventSeqResult struct {
	Seq iter.Seq[StreamEvent]
	Err error
}

// Run executes startingAgent with the provided input using the DefaultRunner.
func Run(ctx context.Context, startingAgent *Agent, input string) (*RunResult, error) {
	return DefaultRunner.Run(ctx, startingAgent, input)
}

// RunStreamed runs a workflow starting at the given agent with the provided input using the
// DefaultRunner and returns a streaming result.
func RunStreamed(ctx context.Context, startingAgent *Agent, input string) (*RunResultStreaming, error) {
	return DefaultRunner.RunStreamed(ctx, startingAgent, input)
}

// RunStreamedChan runs a workflow starting at the given agent with the provided input using the
// DefaultRunner and returns channels that yield streaming events and
// the final streaming error. The events channel is closed once
// streaming completes.
func RunStreamedChan(ctx context.Context, startingAgent *Agent, input string) (<-chan StreamEvent, <-chan error, error) {
	return DefaultRunner.runStreamedChan(ctx, startingAgent, InputString(input))
}

// RunInputStreamedChan runs a workflow starting at the given agent with the provided input using the
// DefaultRunner and returns channels that yield streaming events and
// the final streaming error. The events channel is closed once
// streaming completes.
func RunInputStreamedChan(ctx context.Context, startingAgent *Agent, input []TResponseInputItem) (<-chan StreamEvent, <-chan error, error) {
	return DefaultRunner.runStreamedChan(ctx, startingAgent, InputItems(input))
}

// RunStreamedSeq runs a workflow starting at the given agent in streaming
// mode and returns an EventSeqResult containing the sequence of events.
// The sequence is single-use; after iteration, the Err field will hold
// the streaming error, if any.
func RunStreamedSeq(ctx context.Context, startingAgent *Agent, input string) (*EventSeqResult, error) {
	return DefaultRunner.RunStreamedSeq(ctx, startingAgent, input)
}

// RunInputsStreamedSeq runs a workflow starting at the given agent in streaming
// mode and returns an EventSeqResult containing the sequence of events.
// The sequence is single-use; after iteration, the Err field will hold
// the streaming error, if any.
func RunInputsStreamedSeq(ctx context.Context, startingAgent *Agent, input []TResponseInputItem) (*EventSeqResult, error) {
	return DefaultRunner.RunInputStreamedSeq(ctx, startingAgent, input)
}

// RunInputs executes startingAgent with the provided list of input items using the DefaultRunner.
func RunInputs(ctx context.Context, startingAgent *Agent, input []TResponseInputItem) (*RunResult, error) {
	return DefaultRunner.RunInputs(ctx, startingAgent, input)
}

// RunInputsStreamed executes startingAgent with the provided list of input items using the DefaultRunner
// and returns a streaming result.
func RunInputsStreamed(ctx context.Context, startingAgent *Agent, input []TResponseInputItem) (*RunResultStreaming, error) {
	return DefaultRunner.RunInputsStreamed(ctx, startingAgent, input)
}

// RunFromState resumes a workflow from a serialized RunState using the DefaultRunner.
func RunFromState(ctx context.Context, startingAgent *Agent, state RunState) (*RunResult, error) {
	return DefaultRunner.RunFromState(ctx, startingAgent, state)
}

// RunFromStateStreamed resumes a workflow from a serialized RunState in streaming mode
// using the DefaultRunner.
func RunFromStateStreamed(ctx context.Context, startingAgent *Agent, state RunState) (*RunResultStreaming, error) {
	return DefaultRunner.RunFromStateStreamed(ctx, startingAgent, state)
}

// Run a workflow starting at the given agent. The agent will run in a loop until a final
// output is generated.
//
// The loop runs like so:
//  1. The agent is invoked with the given input.
//  2. If there is a final output (i.e. the agent produces something of type Agent.OutputType, the loop terminates.
//  3. If there's a handoff, we run the loop again, with the new agent.
//  4. Else, we run tool calls (if any), and re-run the loop.
//
// In two cases, the agent run may return an error:
//  1. If the MaxTurns is exceeded, a MaxTurnsExceededError is returned.
//  2. If a guardrail tripwire is triggered, a *GuardrailTripwireTriggeredError is returned.
//
// Note that only the first agent's input guardrails are run.
//
// It returns a run result containing all the inputs, guardrail results and the output of the last
// agent. Agents may perform handoffs, so we don't know the specific type of the output.
func (r Runner) Run(ctx context.Context, startingAgent *Agent, input string) (*RunResult, error) {
	return r.run(ctx, startingAgent, InputString(input))
}

// RunStreamed runs a workflow starting at the given agent in streaming mode.
// The returned result object contains a method you can use to stream semantic
// events as they are generated.
//
// The agent will run in a loop until a final output is generated. The loop runs like so:
//  1. The agent is invoked with the given input.
//  2. If there is a final output (i.e. the agent produces something of type Agent.OutputType, the loop terminates.
//  3. If there's a handoff, we run the loop again, with the new agent.
//  4. Else, we run tool calls (if any), and re-run the loop.
//
// In two cases, the agent run may return an error:
//  1. If the MaxTurns is exceeded, a MaxTurnsExceededError is returned.
//  2. If a guardrail tripwire is triggered, a *GuardrailTripwireTriggeredError is returned.
//
// Note that only the first agent's input guardrails are run.
//
// It returns a result object that contains data about the run, as well as a method to stream events.
func (r Runner) RunStreamed(ctx context.Context, startingAgent *Agent, input string) (*RunResultStreaming, error) {
	return r.runStreamed(ctx, startingAgent, InputString(input))
}

// RunInputs executes startingAgent with the provided list of input items using the Runner configuration.
func (r Runner) RunInputs(ctx context.Context, startingAgent *Agent, input []TResponseInputItem) (*RunResult, error) {
	return r.run(ctx, startingAgent, InputItems(input))
}

// RunInputsStreamed executes startingAgent with the provided list of input items using the Runner configuration and returns a streaming result.
func (r Runner) RunInputsStreamed(ctx context.Context, startingAgent *Agent, input []TResponseInputItem) (*RunResultStreaming, error) {
	return r.runStreamed(ctx, startingAgent, InputItems(input))
}

// RunFromState resumes a workflow from a serialized RunState.
func (r Runner) RunFromState(ctx context.Context, startingAgent *Agent, state RunState) (*RunResult, error) {
	resumeRunner, resumeInput, startingTurn, resumeState, err := r.prepareResumeFromState(startingAgent, state)
	if err != nil {
		return nil, err
	}
	return resumeRunner.runWithStartingTurnAndState(ctx, startingAgent, resumeInput, startingTurn, resumeState)
}

// RunFromStateStreamed resumes a workflow from a serialized RunState in streaming mode.
func (r Runner) RunFromStateStreamed(ctx context.Context, startingAgent *Agent, state RunState) (*RunResultStreaming, error) {
	resumeRunner, resumeInput, startingTurn, resumeState, err := r.prepareResumeFromState(startingAgent, state)
	if err != nil {
		return nil, err
	}
	return resumeRunner.runStreamedWithStartingTurnAndState(ctx, startingAgent, resumeInput, startingTurn, resumeState)
}

// RunStreamedChan runs a workflow starting at the given agent in streaming
// mode and returns channels yielding stream events and the final
// streaming error. The events channel is closed when streaming ends.
func (r Runner) RunStreamedChan(ctx context.Context, startingAgent *Agent, input string) (<-chan StreamEvent, <-chan error, error) {
	return r.runStreamedChan(ctx, startingAgent, InputString(input))
}

// RunInputStreamedChan runs a workflow starting at the given agent in streaming
// mode and returns channels yielding stream events and the final
// streaming error. The events channel is closed when streaming ends.
func (r Runner) RunInputStreamedChan(ctx context.Context, startingAgent *Agent, input []TResponseInputItem) (<-chan StreamEvent, <-chan error, error) {
	return r.runStreamedChan(ctx, startingAgent, InputItems(input))
}

// RunStreamedSeq runs a workflow starting at the given agent in streaming
// mode and returns an EventSeqResult containing the sequence of events.
// The sequence is single-use; after iteration, the Err field will hold
// the streaming error, if any.
func (r Runner) RunStreamedSeq(ctx context.Context, startingAgent *Agent, input string) (*EventSeqResult, error) {
	return r.runStreamedSeq(ctx, startingAgent, InputString(input))
}

// RunInputStreamedSeq runs a workflow starting at the given agent in streaming
// mode and returns an EventSeqResult containing the sequence of events.
// The sequence is single-use; after iteration, the Err field will hold
// the streaming error, if any.
func (r Runner) RunInputStreamedSeq(ctx context.Context, startingAgent *Agent, input []TResponseInputItem) (*EventSeqResult, error) {
	return r.runStreamedSeq(ctx, startingAgent, InputItems(input))
}

func (r Runner) runStreamedChan(ctx context.Context, startingAgent *Agent, input Input) (<-chan StreamEvent, <-chan error, error) {
	result, err := r.runStreamed(ctx, startingAgent, input)
	if err != nil {
		return nil, nil, err
	}

	events := make(chan StreamEvent)
	errs := make(chan error, 1)
	go func() {
		defer close(events)
		defer close(errs)
		errs <- result.StreamEvents(func(event StreamEvent) error {
			events <- event
			return nil
		})
	}()

	return events, errs, nil
}

func (r Runner) runStreamedSeq(ctx context.Context, startingAgent *Agent, input Input) (*EventSeqResult, error) {
	result, err := r.runStreamed(ctx, startingAgent, input)
	if err != nil {
		return nil, err
	}

	res := &EventSeqResult{}
	res.Seq = func(yield func(StreamEvent) bool) {
		res.Err = result.StreamEvents(func(event StreamEvent) error {
			if yield(event) {
				return nil
			}
			// Stop streaming early if the consumer stops iterating.
			result.Cancel()
			return nil
		})
	}
	return res, nil
}

func (r Runner) prepareResumeFromState(
	startingAgent *Agent,
	state RunState,
) (Runner, Input, uint64, *RunState, error) {
	if startingAgent == nil {
		return Runner{}, nil, 0, nil, fmt.Errorf("startingAgent must not be nil")
	}
	if err := state.Validate(); err != nil {
		return Runner{}, nil, 0, nil, err
	}
	if state.CurrentAgentName != "" && state.CurrentAgentName != startingAgent.Name {
		return Runner{}, nil, 0, nil, UserErrorf(
			"run state current agent %q does not match provided starting agent %q",
			state.CurrentAgentName,
			startingAgent.Name,
		)
	}
	if err := state.ApplyStoredToolApprovals(); err != nil {
		return Runner{}, nil, 0, nil, err
	}

	resumeRunner := r
	resumeRunner.Config = state.ResumeRunConfig(r.Config)
	return resumeRunner, state.ResumeInput(), state.CurrentTurn, &state, nil
}

func (r Runner) run(ctx context.Context, startingAgent *Agent, input Input) (*RunResult, error) {
	return r.runWithStartingTurnAndState(ctx, startingAgent, input, 0, nil)
}

func (r Runner) runWithStartingTurn(
	ctx context.Context,
	startingAgent *Agent,
	input Input,
	startingTurn uint64,
) (*RunResult, error) {
	return r.runWithStartingTurnAndState(ctx, startingAgent, input, startingTurn, nil)
}

func (r Runner) runWithStartingTurnAndState(
	ctx context.Context,
	startingAgent *Agent,
	input Input,
	startingTurn uint64,
	resumeState *RunState,
) (*RunResult, error) {
	if startingAgent == nil {
		return nil, fmt.Errorf("startingAgent must not be nil")
	}

	conversationID, previousResponseID, autoPrevious := resolveConversationSettings(r.Config, resumeState)
	if err := validateSessionConversationSettings(r.Config.Session, conversationID, previousResponseID, autoPrevious); err != nil {
		return nil, err
	}
	var serverConversationTracker *OpenAIServerConversationTracker
	if conversationID != "" || autoPrevious {
		serverConversationTracker = NewOpenAIServerConversationTracker(
			conversationID,
			previousResponseID,
			autoPrevious,
		)
		if resumeState != nil {
			sessionItems := runItemsToInputItems(resumeState.SessionItems)
			serverConversationTracker.HydrateFromState(
				InputItems(resumeState.OriginalInput),
				resumeState.GeneratedRunItems,
				resumeState.ModelResponses,
				sessionItems,
			)
		}
	}

	// Prepare input with session if enabled
	preparedInput, err := r.prepareInputWithSession(ctx, input, resumeState != nil)
	if err != nil {
		return nil, err
	}

	hooks := r.Config.Hooks
	if hooks == nil {
		hooks = NoOpRunHooks{}
	}

	toolUseTracker := NewAgentToolUseTracker()
	contextWrapper := NewRunContextWrapper[any](nil)
	_, hasContextOverride := RunContextValueFromContext(ctx)
	if hasContextOverride {
		value, _ := RunContextValueFromContext(ctx)
		contextWrapper.Context = value
	}
	if resumeState != nil {
		// Reset persisted count for the resumed turn and restore approval decisions.
		resumeState.CurrentTurnPersistedItemCount = 0
		resumeState.ApplyToolApprovalsToContext(contextWrapper)
		if resumeState.Context != nil {
			if !hasContextOverride {
				contextWrapper.Context = resumeState.Context.Context
			}
			if resumeState.Context.ToolInput != nil && contextWrapper.ToolInput == nil {
				contextWrapper.ToolInput = resumeState.Context.ToolInput
			}
			if resumeState.Context.Usage != nil {
				contextWrapper.Usage = cloneUsage(resumeState.Context.Usage)
				if u, ok := usage.FromContext(ctx); !ok || u == nil {
					ctx = usage.NewContext(ctx, contextWrapper.Usage)
				}
			}
		}
		if len(resumeState.ToolUseTracker) > 0 {
			toolUseTracker.LoadSnapshot(resumeState.ToolUseTracker)
		}
	}

	var runResult *RunResult

	traceParams := tracing.TraceParams{
		WorkflowName: cmp.Or(r.Config.WorkflowName, DefaultWorkflowName),
		TraceID:      r.Config.TraceID,
		GroupID:      r.Config.GroupID,
		Metadata:     r.Config.TraceMetadata,
		Disabled:     r.Config.TracingDisabled,
	}
	err = ManageTraceCtx(ctx, traceParams, func(ctx context.Context) (err error) {
		currentTurn := startingTurn
		originalInput := CopyInput(preparedInput)

		maxTurns := r.Config.MaxTurns
		if maxTurns == 0 {
			maxTurns = DefaultMaxTurns
		}

		var (
			generatedItems             []RunItem
			modelResponses             []ModelResponse
			inputGuardrailResults      []InputGuardrailResult
			outputGuardrailResults     []OutputGuardrailResult
			toolInputGuardrailResults  []ToolInputGuardrailResult
			toolOutputGuardrailResults []ToolOutputGuardrailResult
			interruptions              []ToolApprovalItem
			currentSpan                tracing.Span
		)
		if resumeState != nil {
			inputGuardrailResults = resumeState.ResumeInputGuardrailResults()
			outputGuardrailResults = resumeState.ResumeOutputGuardrailResults()
			toolInputGuardrailResults = resumeState.ResumeToolInputGuardrailResults()
			toolOutputGuardrailResults = resumeState.ResumeToolOutputGuardrailResults()
		}

		if u, ok := usage.FromContext(ctx); !ok || u == nil {
			if contextWrapper != nil && contextWrapper.Usage != nil {
				ctx = usage.NewContext(ctx, contextWrapper.Usage)
			} else {
				ctx = usage.NewContext(ctx, usage.NewUsage())
				if contextWrapper != nil {
					if u2, ok := usage.FromContext(ctx); ok {
						contextWrapper.Usage = u2
					}
				}
			}
		} else if contextWrapper != nil {
			contextWrapper.Usage = u
		}

		currentAgent := startingAgent
		shouldRunAgentStartHooks := true

		if resumeState != nil && len(resumeState.Interruptions) > 0 {
			pendingInterruptions := slices.Clone(resumeState.Interruptions)
			resolvedItems, pendingInterruptions, toolInputResults, toolOutputResults, functionResults, err := r.resolveFunctionToolInterruptionsOnResume(
				ctx,
				currentAgent,
				resumeState,
				pendingInterruptions,
				hooks,
				r.Config,
				contextWrapper,
			)
			if err != nil {
				return err
			}
			if len(resolvedItems) > 0 {
				generatedItems = append(generatedItems, resolvedItems...)
			}
			if len(toolInputResults) > 0 {
				toolInputGuardrailResults = append(toolInputGuardrailResults, toolInputResults...)
			}
			if len(toolOutputResults) > 0 {
				toolOutputGuardrailResults = append(toolOutputGuardrailResults, toolOutputResults...)
			}

			resolvedItems, pendingInterruptions, err = r.resolveApplyPatchInterruptionsOnResume(
				ctx,
				currentAgent,
				resumeState,
				pendingInterruptions,
				hooks,
				r.Config,
			)
			if err != nil {
				return err
			}
			if len(resolvedItems) > 0 {
				generatedItems = append(generatedItems, resolvedItems...)
			}
			pendingInterruptions = filterPendingMCPInterruptions(pendingInterruptions, contextWrapper)
			if len(pendingInterruptions) > 0 {
				interruptions = slices.Clone(pendingInterruptions)
				runResult = &RunResult{
					Input:                      originalInput,
					NewItems:                   generatedItems,
					RawResponses:               modelResponses,
					InputGuardrailResults:      inputGuardrailResults,
					OutputGuardrailResults:     outputGuardrailResults,
					ToolInputGuardrailResults:  toolInputGuardrailResults,
					ToolOutputGuardrailResults: toolOutputGuardrailResults,
					Interruptions:              interruptions,
					LastAgent:                  currentAgent,
				}
				if err := r.saveResultToSession(ctx, input, runResult, resumeState); err != nil {
					return err
				}
				return nil
			}

			if len(functionResults) > 0 {
				checkToolUse, err := RunImpl().checkForFinalOutputFromTools(ctx, currentAgent, functionResults)
				if err != nil {
					return err
				}
				if checkToolUse.IsFinalOutput {
					if !checkToolUse.FinalOutput.Valid() {
						Logger().Error("Model returned a final output of None. Not raising an error because we assume you know what you're doing.")
					}
					finalOutput := checkToolUse.FinalOutput.Or(nil)
					if currentAgent.OutputType == nil || currentAgent.OutputType.IsPlainText() {
						if _, ok := finalOutput.(string); !ok {
							finalOutput = fmt.Sprintf("%v", finalOutput)
						}
					}

					outputGuardrailResults, err = r.runOutputGuardrails(
						ctx,
						slices.Concat(currentAgent.OutputGuardrails, r.Config.OutputGuardrails),
						currentAgent,
						finalOutput,
					)
					if err != nil {
						return err
					}
					runResult = &RunResult{
						Input:                      originalInput,
						NewItems:                   generatedItems,
						RawResponses:               modelResponses,
						FinalOutput:                finalOutput,
						InputGuardrailResults:      inputGuardrailResults,
						OutputGuardrailResults:     outputGuardrailResults,
						ToolInputGuardrailResults:  toolInputGuardrailResults,
						ToolOutputGuardrailResults: toolOutputGuardrailResults,
						Interruptions:              interruptions,
						LastAgent:                  currentAgent,
					}
					if err := r.saveResultToSession(ctx, input, runResult, resumeState); err != nil {
						return err
					}
					return nil
				}
			}
		}

		defer func() {
			if err != nil {
				var agentsErr *AgentsError
				if errors.As(err, &agentsErr) {
					agentsErr.RunData = &RunErrorDetails{
						Context:                    ctx,
						Input:                      originalInput,
						NewItems:                   generatedItems,
						RawResponses:               modelResponses,
						LastAgent:                  currentAgent,
						InputGuardrailResults:      inputGuardrailResults,
						OutputGuardrailResults:     outputGuardrailResults,
						ToolInputGuardrailResults:  toolInputGuardrailResults,
						ToolOutputGuardrailResults: toolOutputGuardrailResults,
						Interruptions:              interruptions,
					}
				}
			}

			if currentSpan != nil {
				if e := currentSpan.Finish(ctx, true); e != nil {
					err = errors.Join(err, e)
				}
			}
		}()

		childCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		for {
			allTools, err := r.getAllTools(childCtx, currentAgent)
			if err != nil {
				return err
			}

			// Start an agent span if we don't have one. This span is ended if the current
			// agent changes, or if the agent loop ends.
			if currentSpan == nil {
				handoffs, err := r.getHandoffs(ctx, currentAgent)
				if err != nil {
					return err
				}
				handoffNames := make([]string, len(handoffs))
				for i, handoff := range handoffs {
					handoffNames[i] = handoff.AgentName
				}
				outputTypeName := "string"
				if currentAgent.OutputType != nil {
					outputTypeName = currentAgent.OutputType.Name()
				}

				currentSpan = tracing.NewAgentSpan(ctx, tracing.AgentSpanParams{
					Name:       currentAgent.Name,
					Handoffs:   handoffNames,
					OutputType: outputTypeName,
				})
				err = currentSpan.Start(ctx, true)
				if err != nil {
					return err
				}
				toolNames := make([]string, len(allTools))
				for i, tool := range allTools {
					toolNames[i] = tool.ToolName()
				}
				currentSpan.SpanData().(*tracing.AgentSpanData).Tools = toolNames
			}

			currentTurn += 1
			if currentTurn > maxTurns {
				AttachErrorToSpan(currentSpan, tracing.SpanError{
					Message: "Max turns exceeded",
					Data:    map[string]any{"max_turns": maxTurns},
				})
				return MaxTurnsExceededErrorf("max turns %d exceeded", maxTurns)
			}
			Logger().Debug(
				"Running agent",
				slog.String("agentName", currentAgent.Name),
				slog.Uint64("turn", currentTurn),
			)

			var turnResult *SingleStepResult

			if currentTurn == 1 && resumeState == nil {
				var wg sync.WaitGroup
				wg.Add(2)

				var guardrailsError error
				go func() {
					defer wg.Done()
					inputGuardrailResults, guardrailsError = r.runInputGuardrails(
						childCtx,
						startingAgent,
						slices.Concat(startingAgent.InputGuardrails, r.Config.InputGuardrails),
						CopyInput(preparedInput),
					)
					if guardrailsError != nil {
						cancel()
					}
				}()

				var turnError error
				go func() {
					defer wg.Done()
					turnResult, turnError = r.runSingleTurn(
						childCtx,
						currentAgent,
						allTools,
						originalInput,
						generatedItems,
						hooks,
						r.Config,
						shouldRunAgentStartHooks,
						toolUseTracker,
						contextWrapper,
						serverConversationTracker,
						r.Config.PreviousResponseID,
					)
					if turnError != nil {
						cancel()
					}
				}()

				wg.Wait()
				if err = errors.Join(turnError, guardrailsError); err != nil {
					return err
				}
			} else {
				turnResult, err = r.runSingleTurn(
					childCtx,
					currentAgent,
					allTools,
					originalInput,
					generatedItems,
					hooks,
					r.Config,
					shouldRunAgentStartHooks,
					toolUseTracker,
					contextWrapper,
					serverConversationTracker,
					r.Config.PreviousResponseID,
				)
				if err != nil {
					return err
				}
			}

			shouldRunAgentStartHooks = false

			modelResponses = append(modelResponses, turnResult.ModelResponse)
			originalInput = turnResult.OriginalInput
			generatedItems = turnResult.GeneratedItems()
			toolInputGuardrailResults = append(toolInputGuardrailResults, turnResult.ToolInputGuardrailResults...)
			toolOutputGuardrailResults = append(toolOutputGuardrailResults, turnResult.ToolOutputGuardrailResults...)

			switch nextStep := turnResult.NextStep.(type) {
			case NextStepFinalOutput:
				outputGuardrailResults, err = r.runOutputGuardrails(
					childCtx,
					slices.Concat(currentAgent.OutputGuardrails, r.Config.OutputGuardrails),
					currentAgent,
					nextStep.Output,
				)
				if err != nil {
					return err
				}
				runResult = &RunResult{
					Input:                      originalInput,
					NewItems:                   generatedItems,
					RawResponses:               modelResponses,
					FinalOutput:                nextStep.Output,
					InputGuardrailResults:      inputGuardrailResults,
					OutputGuardrailResults:     outputGuardrailResults,
					ToolInputGuardrailResults:  toolInputGuardrailResults,
					ToolOutputGuardrailResults: toolOutputGuardrailResults,
					Interruptions:              interruptions,
					LastAgent:                  currentAgent,
				}
				applyConversationTracking(runResult, serverConversationTracker)

				// Save the conversation to session if enabled
				err = r.saveResultToSession(ctx, input, runResult, resumeState)
				if err != nil {
					return err
				}

				return nil
			case NextStepHandoff:
				currentAgent = nextStep.NewAgent
				err = currentSpan.Finish(ctx, true)
				if err != nil {
					return err
				}
				currentSpan = nil
				shouldRunAgentStartHooks = true
			case NextStepRunAgain:
				// Nothing to do
			case NextStepInterruption:
				interruptions = slices.Clone(nextStep.Interruptions)
				runResult = &RunResult{
					Input:                      originalInput,
					NewItems:                   generatedItems,
					RawResponses:               modelResponses,
					InputGuardrailResults:      inputGuardrailResults,
					OutputGuardrailResults:     outputGuardrailResults,
					ToolInputGuardrailResults:  toolInputGuardrailResults,
					ToolOutputGuardrailResults: toolOutputGuardrailResults,
					Interruptions:              interruptions,
					LastAgent:                  currentAgent,
				}
				applyConversationTracking(runResult, serverConversationTracker)

				err = r.saveResultToSession(ctx, input, runResult, resumeState)
				if err != nil {
					return err
				}

				return nil
			default:
				// This would be an unrecoverable implementation bug, so a panic is appropriate.
				panic(fmt.Errorf("unexpected NextStep type %T", nextStep))
			}
		}
	})
	return runResult, err
}

func (r Runner) resolveApplyPatchInterruptionsOnResume(
	ctx context.Context,
	agent *Agent,
	resumeState *RunState,
	interruptions []ToolApprovalItem,
	hooks RunHooks,
	runConfig RunConfig,
) ([]RunItem, []ToolApprovalItem, error) {
	if resumeState == nil || len(interruptions) == 0 {
		return nil, interruptions, nil
	}

	allTools, err := r.getAllTools(ctx, agent)
	if err != nil {
		return nil, nil, err
	}

	var applyPatchTool *ApplyPatchTool
	for _, tool := range allTools {
		if t, ok := tool.(ApplyPatchTool); ok {
			applyPatchTool = &t
			break
		}
	}
	if applyPatchTool == nil || !hasApplyPatchInterruptions(interruptions, applyPatchTool) {
		return nil, interruptions, nil
	}

	if len(resumeState.ModelResponses) == 0 {
		return nil, interruptions, nil
	}

	handoffs, err := r.getHandoffs(ctx, agent)
	if err != nil {
		return nil, nil, err
	}

	lastResponse := resumeState.ModelResponses[len(resumeState.ModelResponses)-1]
	processed, err := RunImpl().ProcessModelResponse(ctx, agent, allTools, lastResponse, handoffs)
	if err != nil {
		return nil, nil, err
	}

	applyPatchCalls := filterApplyPatchCallsWithoutOutputs(processed.ApplyPatchCalls, resumeState.GeneratedItems)
	if len(applyPatchCalls) == 0 {
		return nil, interruptions, nil
	}

	contextWrapper := NewRunContextWrapper[any](nil)
	resumeState.ApplyToolApprovalsToContext(contextWrapper)

	results, interruptions, err := RunImpl().ExecuteApplyPatchCalls(
		ctx,
		agent,
		applyPatchCalls,
		hooks,
		contextWrapper,
		runConfig,
	)
	if err != nil {
		return nil, nil, err
	}

	pending := filterNonApplyPatchInterruptions(interruptions, applyPatchTool)
	if len(interruptions) > 0 {
		pending = append(pending, interruptions...)
	}

	return results, pending, nil
}

func (r Runner) resolveFunctionToolInterruptionsOnResume(
	ctx context.Context,
	agent *Agent,
	resumeState *RunState,
	interruptions []ToolApprovalItem,
	hooks RunHooks,
	runConfig RunConfig,
	contextWrapper *RunContextWrapper[any],
) ([]RunItem, []ToolApprovalItem, []ToolInputGuardrailResult, []ToolOutputGuardrailResult, []FunctionToolResult, error) {
	if resumeState == nil || len(interruptions) == 0 {
		return nil, interruptions, nil, nil, nil, nil
	}

	allTools, err := r.getAllTools(ctx, agent)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	functionToolNames := make(map[string]struct{})
	for _, tool := range allTools {
		if t, ok := tool.(FunctionTool); ok {
			functionToolNames[t.Name] = struct{}{}
		}
	}
	if len(functionToolNames) == 0 {
		return nil, interruptions, nil, nil, nil, nil
	}

	hasFunctionInterruptions := false
	for _, interruption := range interruptions {
		if _, ok := functionToolNames[resolveApprovalToolName(interruption)]; ok {
			hasFunctionInterruptions = true
			break
		}
	}
	if !hasFunctionInterruptions {
		return nil, interruptions, nil, nil, nil, nil
	}

	if len(resumeState.ModelResponses) == 0 {
		return nil, interruptions, nil, nil, nil, nil
	}

	handoffs, err := r.getHandoffs(ctx, agent)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	lastResponse := resumeState.ModelResponses[len(resumeState.ModelResponses)-1]
	processed, err := RunImpl().ProcessModelResponse(ctx, agent, allTools, lastResponse, handoffs)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	functionCalls := filterFunctionCallsWithoutOutputs(processed.Functions, resumeState.GeneratedItems)
	if len(functionCalls) == 0 {
		return nil, interruptions, nil, nil, nil, nil
	}

	if contextWrapper == nil {
		contextWrapper = NewRunContextWrapper[any](nil)
	}
	resumeState.ApplyToolApprovalsToContext(contextWrapper)

	functionResults, toolInputResults, toolOutputResults, pendingInterruptions, err := RunImpl().ExecuteFunctionToolCalls(
		ctx,
		agent,
		functionCalls,
		hooks,
		contextWrapper,
		runConfig,
	)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	resolvedItems := make([]RunItem, 0, len(functionResults))
	for _, result := range functionResults {
		if result.RunItem == nil {
			continue
		}
		resolvedItems = append(resolvedItems, result.RunItem)
	}

	pending := filterNonFunctionToolInterruptions(interruptions, functionToolNames)
	if len(pendingInterruptions) > 0 {
		pending = append(pending, pendingInterruptions...)
	}

	return resolvedItems, pending, toolInputResults, toolOutputResults, functionResults, nil
}

func filterApplyPatchCallsWithoutOutputs(
	calls []ToolRunApplyPatchCall,
	generatedItems []TResponseInputItem,
) []ToolRunApplyPatchCall {
	if len(calls) == 0 {
		return nil
	}
	out := make([]ToolRunApplyPatchCall, 0, len(calls))
	for _, call := range calls {
		callID, err := extractApplyPatchCallID(call.ToolCall)
		if err != nil || callID == "" {
			continue
		}
		if applyPatchOutputExists(callID, generatedItems) {
			continue
		}
		out = append(out, call)
	}
	return out
}

func filterFunctionCallsWithoutOutputs(
	calls []ToolRunFunction,
	generatedItems []TResponseInputItem,
) []ToolRunFunction {
	if len(calls) == 0 {
		return nil
	}
	out := make([]ToolRunFunction, 0, len(calls))
	for _, call := range calls {
		callID := call.ToolCall.CallID
		if callID == "" {
			continue
		}
		if functionCallOutputExists(callID, generatedItems) {
			continue
		}
		out = append(out, call)
	}
	return out
}

func applyPatchOutputExists(callID string, items []TResponseInputItem) bool {
	for _, item := range items {
		itemType := item.GetType()
		if itemType == nil || *itemType != "apply_patch_call_output" {
			continue
		}
		existingID := item.GetCallID()
		if existingID != nil && *existingID == callID {
			return true
		}
	}
	return false
}

func functionCallOutputExists(callID string, items []TResponseInputItem) bool {
	for _, item := range items {
		itemType := item.GetType()
		if itemType == nil || *itemType != "function_call_output" {
			continue
		}
		existingID := item.GetCallID()
		if existingID != nil && *existingID == callID {
			return true
		}
	}
	return false
}

func hasApplyPatchInterruptions(interruptions []ToolApprovalItem, tool *ApplyPatchTool) bool {
	for _, interruption := range interruptions {
		if isApplyPatchName(resolveApprovalToolName(interruption), tool) {
			return true
		}
	}
	return false
}

func filterNonFunctionToolInterruptions(
	interruptions []ToolApprovalItem,
	functionToolNames map[string]struct{},
) []ToolApprovalItem {
	if len(interruptions) == 0 {
		return nil
	}
	out := make([]ToolApprovalItem, 0, len(interruptions))
	for _, interruption := range interruptions {
		if _, ok := functionToolNames[resolveApprovalToolName(interruption)]; ok {
			continue
		}
		out = append(out, interruption)
	}
	return out
}

func filterPendingMCPInterruptions(
	interruptions []ToolApprovalItem,
	contextWrapper *RunContextWrapper[any],
) []ToolApprovalItem {
	if len(interruptions) == 0 {
		return nil
	}
	if contextWrapper == nil {
		return interruptions
	}
	out := make([]ToolApprovalItem, 0, len(interruptions))
	for _, interruption := range interruptions {
		if !isMCPApprovalItem(interruption) {
			out = append(out, interruption)
			continue
		}
		callID := resolveApprovalCallID(interruption)
		if callID == "" {
			out = append(out, interruption)
			continue
		}
		_, known := contextWrapper.GetApprovalStatus(
			resolveApprovalToolName(interruption),
			callID,
			&interruption,
		)
		if known {
			continue
		}
		out = append(out, interruption)
	}
	return out
}

func filterNonApplyPatchInterruptions(
	interruptions []ToolApprovalItem,
	tool *ApplyPatchTool,
) []ToolApprovalItem {
	if len(interruptions) == 0 {
		return nil
	}
	out := make([]ToolApprovalItem, 0, len(interruptions))
	for _, interruption := range interruptions {
		if isApplyPatchName(resolveApprovalToolName(interruption), tool) {
			continue
		}
		out = append(out, interruption)
	}
	return out
}

func (r Runner) runStreamed(ctx context.Context, startingAgent *Agent, input Input) (*RunResultStreaming, error) {
	return r.runStreamedWithStartingTurnAndState(ctx, startingAgent, input, 0, nil)
}

func (r Runner) runStreamedWithStartingTurn(
	ctx context.Context,
	startingAgent *Agent,
	input Input,
	startingTurn uint64,
) (*RunResultStreaming, error) {
	return r.runStreamedWithStartingTurnAndState(ctx, startingAgent, input, startingTurn, nil)
}

func (r Runner) runStreamedWithStartingTurnAndState(
	ctx context.Context,
	startingAgent *Agent,
	input Input,
	startingTurn uint64,
	resumeState *RunState,
) (*RunResultStreaming, error) {
	if startingAgent == nil {
		return nil, fmt.Errorf("startingAgent must not be nil")
	}

	conversationID, previousResponseID, autoPrevious := resolveConversationSettings(r.Config, resumeState)
	if err := validateSessionConversationSettings(r.Config.Session, conversationID, previousResponseID, autoPrevious); err != nil {
		return nil, err
	}

	maxTurns := r.Config.MaxTurns
	if maxTurns == 0 {
		maxTurns = DefaultMaxTurns
	}

	hooks := r.Config.Hooks
	if hooks == nil {
		hooks = NoOpRunHooks{}
	}

	// If there's already a trace, we don't create a new one. In addition, we can't end the trace
	// here, because the actual work is done in StreamEvents and this method ends before that.
	var newTrace tracing.Trace
	if tracing.GetCurrentTrace(ctx) == nil {
		ctx = tracing.ContextWithClonedOrNewScope(ctx)
		newTrace = tracing.NewTrace(ctx, tracing.TraceParams{
			WorkflowName: cmp.Or(r.Config.WorkflowName, DefaultWorkflowName),
			TraceID:      r.Config.TraceID,
			GroupID:      r.Config.GroupID,
			Metadata:     r.Config.TraceMetadata,
			Disabled:     r.Config.TracingDisabled,
		})
	}

	if resumeState != nil && resumeState.Context != nil && resumeState.Context.Usage != nil {
		if u, ok := usage.FromContext(ctx); !ok || u == nil {
			ctx = usage.NewContext(ctx, cloneUsage(resumeState.Context.Usage))
		}
	}

	if resumeState != nil {
		Logger().Debug(
			"Resuming from RunState in run_streaming()",
			buildResumedStreamDebugAttrs(resumeState, !DontLogToolData)...,
		)
	}

	if u, ok := usage.FromContext(ctx); !ok || u == nil {
		ctx = usage.NewContext(ctx, usage.NewUsage())
	}

	streamedResult := newRunResultStreaming(ctx)
	streamedResult.setInput(CopyInput(input))
	streamedResult.setCurrentAgent(startingAgent)
	streamedResult.setCurrentTurn(startingTurn)
	streamedResult.setMaxTurns(maxTurns)
	streamedResult.setCurrentAgentOutputType(startingAgent.OutputType)
	streamedResult.setTrace(newTrace)
	if resumeState != nil {
		streamedResult.setInputGuardrailResults(resumeState.ResumeInputGuardrailResults())
		streamedResult.setOutputGuardrailResults(resumeState.ResumeOutputGuardrailResults())
		streamedResult.setToolInputGuardrailResults(resumeState.ResumeToolInputGuardrailResults())
		streamedResult.setToolOutputGuardrailResults(resumeState.ResumeToolOutputGuardrailResults())
	}

	// Kick off the actual agent loop in the background and return the streamed result object.
	streamedResult.createRunImplTask(ctx, func(ctx context.Context) error {
		return r.startStreaming(
			ctx,
			input,
			streamedResult,
			startingAgent,
			startingTurn,
			maxTurns,
			hooks,
			r.Config,
			r.Config.PreviousResponseID,
			resumeState,
		)
	})

	return streamedResult, nil
}

func resolveConversationSettings(
	runConfig RunConfig,
	runState *RunState,
) (conversationID string, previousResponseID string, autoPrevious bool) {
	conversationID = runConfig.ConversationID
	previousResponseID = runConfig.PreviousResponseID
	autoPrevious = runConfig.AutoPreviousResponseID

	if runState != nil {
		if conversationID == "" && runState.ConversationID != "" {
			conversationID = runState.ConversationID
		}
		if previousResponseID == "" && runState.PreviousResponseID != "" {
			previousResponseID = runState.PreviousResponseID
		}
		if !autoPrevious && runState.AutoPreviousResponseID {
			autoPrevious = true
		}
		runState.ConversationID = conversationID
		runState.PreviousResponseID = previousResponseID
		runState.AutoPreviousResponseID = autoPrevious
	}

	return conversationID, previousResponseID, autoPrevious
}

func validateSessionConversationSettings(
	session memory.Session,
	conversationID string,
	previousResponseID string,
	autoPrevious bool,
) error {
	if session == nil {
		return nil
	}
	if conversationID == "" && previousResponseID == "" && !autoPrevious {
		return nil
	}
	return NewUserError(
		"Session persistence cannot be combined with conversation_id, " +
			"previous_response_id, or auto_previous_response_id.",
	)
}

func applyConversationTracking(result *RunResult, tracker *OpenAIServerConversationTracker) {
	if result == nil || tracker == nil {
		return
	}
	result.ConversationID = tracker.ConversationID
	result.PreviousResponseID = tracker.PreviousResponseID
	result.AutoPreviousResponseID = tracker.AutoPreviousResponseID
}

// Apply optional CallModelInputFilter to modify model input.
//
// Returns a ModelInputData that will be sent to the model.
func (r Runner) maybeFilterModelInput(
	ctx context.Context,
	agent *Agent,
	runConfig RunConfig,
	inputItems []TResponseInputItem,
	systemInstructions param.Opt[string],
) (_ *ModelInputData, err error) {
	effectiveInstructions := systemInstructions
	effectiveInput := inputItems

	if runConfig.CallModelInputFilter == nil {
		return &ModelInputData{
			Input:        effectiveInput,
			Instructions: effectiveInstructions,
		}, nil
	}

	defer func() {
		if err != nil {
			AttachErrorToCurrentSpan(ctx, tracing.SpanError{
				Message: "Error in CallModelInputFilter",
				Data:    map[string]any{"error": err.Error()},
			})
		}
	}()

	modelInput := ModelInputData{
		Input:        slices.Clone(effectiveInput),
		Instructions: effectiveInstructions,
	}

	filterPayload := CallModelData{
		ModelData: modelInput,
		Agent:     agent,
	}

	updated, err := runConfig.CallModelInputFilter(ctx, filterPayload)
	if err != nil {
		return nil, err
	}
	if updated == nil {
		return nil, fmt.Errorf("CallModelInputFilter returned nil *ModelInputData but no error")
	}
	return updated, nil
}

func (r Runner) runInputGuardrailsWithQueue(
	ctx context.Context,
	agent *Agent,
	guardrails []InputGuardrail,
	input Input,
	streamedResult *RunResultStreaming,
	parentSpan tracing.Span,
) error {
	queue := streamedResult.inputGuardrailQueue

	if len(guardrails) == 0 {
		return nil
	}

	guardrailResults := make([]InputGuardrailResult, 0, len(guardrails))
	guardrailErrors := make([]error, 0, len(guardrails))

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var mu sync.Mutex

	var wg sync.WaitGroup
	wg.Add(len(guardrails))

	// We'll run the guardrails and push them onto the queue as they complete
	for _, guardrail := range guardrails {
		guardrail := guardrail
		go func() {
			defer wg.Done()

			result, err := RunImpl().RunSingleInputGuardrail(childCtx, agent, guardrail, input)
			if err != nil {
				cancel()
				mu.Lock()
				guardrailErrors = append(guardrailErrors, fmt.Errorf("failed to run input guardrail %s: %w", guardrail.Name, err))
				mu.Unlock()
				return
			}

			mu.Lock()
			guardrailResults = append(guardrailResults, result)
			snapshot := slices.Clone(guardrailResults)
			mu.Unlock()
			streamedResult.setInputGuardrailResults(snapshot)

			queue.Put(result)

			if result.Output.TripwireTriggered {
				mu.Lock()
				defer mu.Unlock()
				AttachErrorToSpan(parentSpan, tracing.SpanError{
					Message: "Guardrail tripwire triggered",
					Data: map[string]any{
						"guardrail": result.Guardrail.Name,
						"type":      "input_guardrail",
					},
				})
			}
		}()
	}

	wg.Wait()
	if err := errors.Join(guardrailErrors...); err != nil {
		return err
	}

	streamedResult.setInputGuardrailResults(guardrailResults)
	return nil
}

func (r Runner) startStreaming(
	ctx context.Context,
	startingInput Input,
	streamedResult *RunResultStreaming,
	startingAgent *Agent,
	startingTurn uint64,
	maxTurns uint64,
	hooks RunHooks,
	runConfig RunConfig,
	previousResponseID string,
	resumeState *RunState,
) (err error) {
	currentAgent := startingAgent
	var currentSpan tracing.Span

	defer func() {
		// Recover from panics to ensure the queue is properly closed
		if r := recover(); r != nil {
			err = errors.Join(err, fmt.Errorf("startStreaming panicked: %v", r))
		}

		if err != nil {
			var agentsErr *AgentsError
			if errors.As(err, &agentsErr) {
				agentsErr.RunData = &RunErrorDetails{
					Context:                    ctx,
					Input:                      streamedResult.Input(),
					NewItems:                   streamedResult.NewItems(),
					RawResponses:               streamedResult.RawResponses(),
					LastAgent:                  currentAgent,
					InputGuardrailResults:      streamedResult.InputGuardrailResults(),
					OutputGuardrailResults:     streamedResult.OutputGuardrailResults(),
					ToolInputGuardrailResults:  streamedResult.ToolInputGuardrailResults(),
					ToolOutputGuardrailResults: streamedResult.ToolOutputGuardrailResults(),
					Interruptions:              streamedResult.Interruptions(),
				}
			} else if currentSpan != nil {
				AttachErrorToSpan(currentSpan, tracing.SpanError{
					Message: "Error in agent run",
					Data:    map[string]any{"error": err.Error()},
				})
			}

			streamedResult.markAsComplete()
			streamedResult.eventQueue.Put(queueCompleteSentinel{})
		}

		if t := streamedResult.getInputGuardrailsTask(); t != nil && !t.IsDone() {
			_ = t.Await()
		}

		if currentSpan != nil {
			if e := currentSpan.Finish(ctx, true); e != nil {
				err = errors.Join(err, e)
			}
		}

		if trace := streamedResult.getTrace(); trace != nil {
			if e := trace.Finish(ctx, true); e != nil {
				err = errors.Join(err, e)
			}
		}
	}()

	if trace := streamedResult.getTrace(); trace != nil {
		err = trace.Start(ctx, true)
		if err != nil {
			return err
		}
	}

	conversationID, resolvedPreviousResponseID, autoPrevious := resolveConversationSettings(runConfig, resumeState)
	previousResponseID = resolvedPreviousResponseID

	var serverConversationTracker *OpenAIServerConversationTracker
	if conversationID != "" || autoPrevious {
		serverConversationTracker = NewOpenAIServerConversationTracker(
			conversationID,
			previousResponseID,
			autoPrevious,
		)
		if resumeState != nil {
			sessionItems := runItemsToInputItems(resumeState.SessionItems)
			serverConversationTracker.HydrateFromState(
				InputItems(resumeState.OriginalInput),
				resumeState.GeneratedRunItems,
				resumeState.ModelResponses,
				sessionItems,
			)
		}
		streamedResult.setConversationID(serverConversationTracker.ConversationID)
		streamedResult.setPreviousResponseID(serverConversationTracker.PreviousResponseID)
		streamedResult.setAutoPreviousResponseID(serverConversationTracker.AutoPreviousResponseID)
	}

	currentTurn := startingTurn
	shouldRunAgentStartHooks := true
	toolUseTracker := NewAgentToolUseTracker()
	isResumedState := resumeState != nil
	contextWrapper := NewRunContextWrapper[any](nil)
	_, hasContextOverride := RunContextValueFromContext(ctx)
	if hasContextOverride {
		value, _ := RunContextValueFromContext(ctx)
		contextWrapper.Context = value
	}
	if resumeState != nil {
		// Reset persisted count for the resumed turn and restore approval decisions.
		resumeState.CurrentTurnPersistedItemCount = 0
		resumeState.ApplyToolApprovalsToContext(contextWrapper)
		if resumeState.Context != nil {
			if !hasContextOverride {
				contextWrapper.Context = resumeState.Context.Context
			}
			if resumeState.Context.ToolInput != nil && contextWrapper.ToolInput == nil {
				contextWrapper.ToolInput = resumeState.Context.ToolInput
			}
			if resumeState.Context.Usage != nil {
				contextWrapper.Usage = cloneUsage(resumeState.Context.Usage)
				if u, ok := usage.FromContext(ctx); !ok || u == nil {
					ctx = usage.NewContext(ctx, contextWrapper.Usage)
				}
			}
		}
		if len(resumeState.ToolUseTracker) > 0 {
			toolUseTracker.LoadSnapshot(resumeState.ToolUseTracker)
		}
	}

	streamedResult.eventQueue.Put(AgentUpdatedStreamEvent{
		NewAgent: currentAgent,
		Type:     "agent_updated_stream_event",
	})

	// Prepare input with session if enabled
	preparedInput, err := r.prepareInputWithSession(ctx, startingInput, isResumedState)
	if err != nil {
		return err
	}

	// Update the streamed result with the prepared input
	streamedResult.setInput(preparedInput)

	for !streamedResult.IsComplete() {
		allTools, err := r.getAllTools(ctx, currentAgent)
		if err != nil {
			return err
		}

		// Start an agent span if we don't have one. This span is ended if the current
		// agent changes, or if the agent loop ends.
		if currentSpan == nil {
			handoffs, err := r.getHandoffs(ctx, currentAgent)
			if err != nil {
				return err
			}
			handoffNames := make([]string, len(handoffs))
			for i, handoff := range handoffs {
				handoffNames[i] = handoff.AgentName
			}
			outputTypeName := "string"
			if currentAgent.OutputType != nil {
				outputTypeName = currentAgent.OutputType.Name()
			}

			currentSpan = tracing.NewAgentSpan(ctx, tracing.AgentSpanParams{
				Name:       currentAgent.Name,
				Handoffs:   handoffNames,
				OutputType: outputTypeName,
			})
			err = currentSpan.Start(ctx, true)
			if err != nil {
				return err
			}
			toolNames := make([]string, len(allTools))
			for i, tool := range allTools {
				toolNames[i] = tool.ToolName()
			}
			currentSpan.SpanData().(*tracing.AgentSpanData).Tools = toolNames
		}

		currentTurn += 1
		streamedResult.setCurrentTurn(currentTurn)

		if currentTurn > maxTurns {
			AttachErrorToSpan(currentSpan, tracing.SpanError{
				Message: "Max turns exceeded",
				Data:    map[string]any{"max_turns": maxTurns},
			})
			streamedResult.eventQueue.Put(queueCompleteSentinel{})
			break
		}

		if currentTurn == 1 && !isResumedState {
			// Run the input guardrails in the background and put the results on the queue
			streamedResult.createInputGuardrailsTask(ctx, func(ctx context.Context) error {
				return r.runInputGuardrailsWithQueue(
					ctx,
					startingAgent,
					slices.Concat(startingAgent.InputGuardrails, runConfig.InputGuardrails),
					InputItems(ItemHelpers().InputToNewInputList(preparedInput)),
					streamedResult,
					currentSpan,
				)
			})
		}

		turnResult, err := r.runSingleTurnStreamed(
			ctx,
			streamedResult,
			currentAgent,
			hooks,
			runConfig,
			shouldRunAgentStartHooks,
			toolUseTracker,
			contextWrapper,
			allTools,
			previousResponseID,
			serverConversationTracker,
		)
		if err != nil {
			return err
		}
		shouldRunAgentStartHooks = false

		streamedResult.appendRawResponses(turnResult.ModelResponse)
		streamedResult.setInput(turnResult.OriginalInput)
		streamedResult.setNewItems(turnResult.GeneratedItems())
		streamedResult.appendToolInputGuardrailResults(turnResult.ToolInputGuardrailResults...)
		streamedResult.appendToolOutputGuardrailResults(turnResult.ToolOutputGuardrailResults...)

		switch nextStep := turnResult.NextStep.(type) {
		case NextStepFinalOutput:
			streamedResult.createOutputGuardrailsTask(ctx, func(ctx context.Context) ([]OutputGuardrailResult, error) {
				return r.runOutputGuardrails(
					ctx,
					slices.Concat(currentAgent.OutputGuardrails, runConfig.OutputGuardrails),
					currentAgent,
					nextStep.Output,
				)
			})

			taskResult := streamedResult.getOutputGuardrailsTask().Await()

			var outputGuardrailResults []OutputGuardrailResult
			if taskResult.Error == nil {
				// Errors will be checked in the stream-events loop
				outputGuardrailResults = taskResult.Value
			}

			streamedResult.setOutputGuardrailResults(outputGuardrailResults)
			streamedResult.setFinalOutput(nextStep.Output)
			streamedResult.markAsComplete()

			// Save the conversation to session if enabled
			// Create a temporary RunResult for session saving
			tempResult := &RunResult{
				Input:                      streamedResult.Input(),
				NewItems:                   streamedResult.NewItems(),
				RawResponses:               streamedResult.RawResponses(),
				FinalOutput:                streamedResult.FinalOutput(),
				InputGuardrailResults:      streamedResult.InputGuardrailResults(),
				OutputGuardrailResults:     streamedResult.OutputGuardrailResults(),
				ToolInputGuardrailResults:  streamedResult.ToolInputGuardrailResults(),
				ToolOutputGuardrailResults: streamedResult.ToolOutputGuardrailResults(),
				Interruptions:              streamedResult.Interruptions(),
				LastAgent:                  currentAgent,
			}
			err = r.saveResultToSession(ctx, startingInput, tempResult, resumeState)
			if err != nil {
				return err
			}

			streamedResult.eventQueue.Put(queueCompleteSentinel{})
		case NextStepHandoff:
			currentAgent = nextStep.NewAgent
			err = currentSpan.Finish(ctx, true)
			if err != nil {
				return err
			}
			currentSpan = nil
			shouldRunAgentStartHooks = true
			streamedResult.eventQueue.Put(AgentUpdatedStreamEvent{
				NewAgent: currentAgent,
				Type:     "agent_updated_stream_event",
			})
		case NextStepRunAgain:
			// Nothing to do
		case NextStepInterruption:
			streamedResult.setInterruptions(slices.Clone(nextStep.Interruptions))
			streamedResult.markAsComplete()

			tempResult := &RunResult{
				Input:                      streamedResult.Input(),
				NewItems:                   streamedResult.NewItems(),
				RawResponses:               streamedResult.RawResponses(),
				InputGuardrailResults:      streamedResult.InputGuardrailResults(),
				OutputGuardrailResults:     streamedResult.OutputGuardrailResults(),
				ToolInputGuardrailResults:  streamedResult.ToolInputGuardrailResults(),
				ToolOutputGuardrailResults: streamedResult.ToolOutputGuardrailResults(),
				Interruptions:              streamedResult.Interruptions(),
				LastAgent:                  currentAgent,
			}
			err = r.saveResultToSession(ctx, startingInput, tempResult, resumeState)
			if err != nil {
				return err
			}

			streamedResult.eventQueue.Put(queueCompleteSentinel{})
		default:
			// This would be an unrecoverable implementation bug, so a panic is appropriate.
			panic(fmt.Errorf("unexpected NextStep type %T", nextStep))
		}
	}

	streamedResult.markAsComplete()
	return nil
}

func (r Runner) runSingleTurnStreamed(
	ctx context.Context,
	streamedResult *RunResultStreaming,
	agent *Agent,
	hooks RunHooks,
	runConfig RunConfig,
	shouldRunAgentStartHooks bool,
	toolUseTracker *AgentToolUseTracker,
	contextWrapper *RunContextWrapper[any],
	allTools []Tool,
	previousResponseID string,
	serverConversationTracker *OpenAIServerConversationTracker,
) (*SingleStepResult, error) {
	if shouldRunAgentStartHooks {
		childCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		var wg sync.WaitGroup
		var hooksErrors [2]error

		wg.Add(1)
		go func() {
			defer wg.Done()
			err := hooks.OnAgentStart(childCtx, agent)
			if err != nil {
				cancel()
				hooksErrors[0] = fmt.Errorf("RunHooks.OnAgentStart failed: %w", err)
			}
		}()

		if agent.Hooks != nil {
			wg.Add(1)
			go func() {
				defer wg.Done()
				err := agent.Hooks.OnStart(childCtx, agent)
				if err != nil {
					cancel()
					hooksErrors[1] = fmt.Errorf("AgentHooks.OnStart failed: %w", err)
				}
			}()
		}

		wg.Wait()
		if err := errors.Join(hooksErrors[:]...); err != nil {
			return nil, err
		}
	}

	streamedResult.setCurrentAgent(agent)
	streamedResult.setCurrentAgentOutputType(agent.OutputType)

	systemPrompt, promptConfig, err := getAgentSystemPromptAndPromptConfig(ctx, agent)
	if err != nil {
		return nil, err
	}

	handoffs, err := r.getHandoffs(ctx, agent)
	if err != nil {
		return nil, err
	}

	model, err := r.getModel(agent, runConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to get model: %w", err)
	}
	emittedToolCallIDs := map[string]struct{}{}
	emittedReasoningIDs := map[string]struct{}{}
	toolCallItemFromOutput := func(output responses.ResponseOutputItemUnion) (ToolCallItem, string, bool) {
		if output.Type != "function_call" {
			return ToolCallItem{}, "", false
		}
		callID := output.CallID
		if callID == "" {
			callID = output.ID
		}
		outputItem := responses.ResponseFunctionToolCall{
			Arguments: output.Arguments,
			CallID:    output.CallID,
			Name:      output.Name,
			Type:      constant.FunctionCall(output.Type),
			ID:        output.ID,
			Status:    responses.ResponseFunctionToolCallStatus(output.Status),
		}
		if outputItem.Type == "" {
			outputItem.Type = constant.ValueOf[constant.FunctionCall]()
		}
		return ToolCallItem{
			Agent:   agent,
			RawItem: ResponseFunctionToolCall(outputItem),
			Type:    "tool_call_item",
		}, callID, callID != ""
	}
	reasoningItemFromOutput := func(output responses.ResponseOutputItemUnion) (ReasoningItem, string, bool) {
		if output.Type != "reasoning" {
			return ReasoningItem{}, "", false
		}
		reasoningID := output.ID
		if reasoningID == "" {
			return ReasoningItem{}, "", false
		}
		outputItem := responses.ResponseReasoningItem{
			ID:               output.ID,
			Summary:          output.Summary,
			Type:             constant.Reasoning(output.Type),
			EncryptedContent: output.EncryptedContent,
			Status:           responses.ResponseReasoningItemStatus(output.Status),
		}
		if outputItem.Type == "" {
			outputItem.Type = constant.ValueOf[constant.Reasoning]()
		}
		return ReasoningItem{
			Agent:   agent,
			RawItem: outputItem,
			Type:    "reasoning_item",
		}, reasoningID, true
	}
	toolCallIDFromItem := func(item ToolCallItemType) string {
		switch v := item.(type) {
		case ResponseFunctionToolCall:
			typed := responses.ResponseFunctionToolCall(v)
			if typed.CallID != "" {
				return typed.CallID
			}
			return typed.ID
		case *ResponseFunctionToolCall:
			if v == nil {
				return ""
			}
			typed := responses.ResponseFunctionToolCall(*v)
			if typed.CallID != "" {
				return typed.CallID
			}
			return typed.ID
		default:
			return ""
		}
	}
	modelSettings := agent.ModelSettings.Resolve(runConfig.ModelSettings)
	modelSettings = RunImpl().MaybeResetToolChoice(agent, toolUseTracker, modelSettings)

	var finalResponse *ModelResponse

	var input []TResponseInputItem
	if serverConversationTracker != nil {
		input = serverConversationTracker.PrepareInput(
			streamedResult.Input(),
			streamedResult.NewItems(),
		)
	} else {
		input = ItemHelpers().InputToNewInputList(streamedResult.Input())
		for _, item := range streamedResult.NewItems() {
			input = append(input, item.ToInputItem())
		}
	}

	filtered, err := r.maybeFilterModelInput(
		ctx,
		agent,
		runConfig,
		input,
		systemPrompt,
	)
	if err != nil {
		return nil, err
	}
	if serverConversationTracker != nil {
		serverConversationTracker.MarkInputAsSent(filtered.Input)
	}

	conversationID := runConfig.ConversationID
	if serverConversationTracker != nil {
		conversationID = serverConversationTracker.ConversationID
		previousResponseID = serverConversationTracker.PreviousResponseID
	}

	// Call hook just before the model is invoked, with the correct system prompt.
	if agent.Hooks != nil {
		err = agent.Hooks.OnLLMStart(ctx, agent, filtered.Instructions, filtered.Input)
		if err != nil {
			return nil, err
		}
	}

	// 1. Stream the output events
	modelResponseParams := ModelResponseParams{
		SystemInstructions: filtered.Instructions,
		Input:              InputItems(filtered.Input),
		ModelSettings:      modelSettings,
		Tools:              allTools,
		OutputType:         agent.OutputType,
		Handoffs:           handoffs,
		Tracing: GetModelTracingImpl(
			runConfig.TracingDisabled,
			runConfig.TraceIncludeSensitiveData.Or(true),
		),
		PreviousResponseID: previousResponseID,
		ConversationID:     conversationID,
		Prompt:             promptConfig,
	}
	err = model.StreamResponse(
		ctx, modelResponseParams,
		func(ctx context.Context, event TResponseStreamEvent) error {
			if event.Type == "response.completed" {
				u := usage.NewUsage()
				if !reflect.ValueOf(event.Response.Usage).IsZero() {
					*u = usage.Usage{
						Requests:            1,
						InputTokens:         uint64(event.Response.Usage.InputTokens),
						InputTokensDetails:  event.Response.Usage.InputTokensDetails,
						OutputTokens:        uint64(event.Response.Usage.OutputTokens),
						OutputTokensDetails: event.Response.Usage.OutputTokensDetails,
						TotalTokens:         uint64(event.Response.Usage.TotalTokens),
					}
				}
				finalResponse = &ModelResponse{
					Output:     event.Response.Output,
					Usage:      u,
					ResponseID: event.Response.ID,
				}
				if contextUsage, _ := usage.FromContext(ctx); contextUsage != nil {
					contextUsage.Add(u)
				}
			}
			streamedResult.eventQueue.Put(RawResponsesStreamEvent{
				Data: event,
				Type: "raw_response_event",
			})
			if event.Type == "response.output_item.done" {
				outputItem := event.Item
				if toolItem, callID, ok := toolCallItemFromOutput(outputItem); ok {
					if _, seen := emittedToolCallIDs[callID]; !seen {
						emittedToolCallIDs[callID] = struct{}{}
						streamedResult.eventQueue.Put(NewRunItemStreamEvent(StreamEventToolCalled, toolItem))
					}
				} else if reasoningItem, reasoningID, ok := reasoningItemFromOutput(outputItem); ok {
					if _, seen := emittedReasoningIDs[reasoningID]; !seen {
						emittedReasoningIDs[reasoningID] = struct{}{}
						streamedResult.eventQueue.Put(NewRunItemStreamEvent(StreamEventReasoningItemCreated, reasoningItem))
					}
				}
			}
			return nil
		},
	)
	if err != nil {
		return nil, err
	}

	// Call hook just after the model response is finalized.
	if agent.Hooks != nil && finalResponse != nil {
		err = agent.Hooks.OnLLMEnd(ctx, agent, *finalResponse)
		if err != nil {
			return nil, err
		}
	}

	// 2. At this point, the streaming is complete for this turn of the agent loop.
	if finalResponse == nil {
		return nil, NewModelBehaviorError("Model did not produce a final response!")
	}

	if serverConversationTracker != nil {
		serverConversationTracker.TrackServerItems(finalResponse)
		streamedResult.setConversationID(serverConversationTracker.ConversationID)
		streamedResult.setPreviousResponseID(serverConversationTracker.PreviousResponseID)
		streamedResult.setAutoPreviousResponseID(serverConversationTracker.AutoPreviousResponseID)
	}

	// 3. Now, we can process the turn as we do in the non-streaming case
	singleStepResult, err := r.getSingleStepResultFromResponse(
		ctx,
		agent,
		allTools,
		streamedResult.Input(),
		streamedResult.NewItems(),
		*finalResponse,
		agent.OutputType,
		handoffs,
		hooks,
		runConfig,
		toolUseTracker,
		contextWrapper,
	)
	if err != nil {
		return nil, err
	}

	streamedStepResult := *singleStepResult
	if len(emittedToolCallIDs) > 0 || len(emittedReasoningIDs) > 0 {
		filtered := make([]RunItem, 0, len(streamedStepResult.NewStepItems))
		for _, item := range streamedStepResult.NewStepItems {
			switch typed := item.(type) {
			case ToolCallItem:
				callID := toolCallIDFromItem(typed.RawItem)
				if callID != "" {
					if _, seen := emittedToolCallIDs[callID]; seen {
						continue
					}
				}
			case *ToolCallItem:
				if typed != nil {
					callID := toolCallIDFromItem(typed.RawItem)
					if callID != "" {
						if _, seen := emittedToolCallIDs[callID]; seen {
							continue
						}
					}
				}
			case ReasoningItem:
				if typed.RawItem.ID != "" {
					if _, seen := emittedReasoningIDs[typed.RawItem.ID]; seen {
						continue
					}
				}
			case *ReasoningItem:
				if typed != nil && typed.RawItem.ID != "" {
					if _, seen := emittedReasoningIDs[typed.RawItem.ID]; seen {
						continue
					}
				}
			}
			filtered = append(filtered, item)
		}
		streamedStepResult.NewStepItems = filtered
	}
	RunImpl().StreamStepResultToQueue(streamedStepResult, streamedResult.eventQueue)
	return singleStepResult, nil
}

func (r Runner) runSingleTurn(
	ctx context.Context,
	agent *Agent,
	allTools []Tool,
	originalInput Input,
	generatedItems []RunItem,
	hooks RunHooks,
	runConfig RunConfig,
	shouldRunAgentStartHooks bool,
	toolUseTracker *AgentToolUseTracker,
	contextWrapper *RunContextWrapper[any],
	serverConversationTracker *OpenAIServerConversationTracker,
	previousResponseID string,
) (*SingleStepResult, error) {
	// Ensure we run the hooks before anything else
	if shouldRunAgentStartHooks {
		childCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		var wg sync.WaitGroup
		var hooksErrors [2]error

		wg.Add(1)
		go func() {
			defer wg.Done()
			err := hooks.OnAgentStart(childCtx, agent)
			if err != nil {
				cancel()
				hooksErrors[0] = fmt.Errorf("RunHooks.OnAgentStart failed: %w", err)
			}
		}()

		if agent.Hooks != nil {
			wg.Add(1)
			go func() {
				defer wg.Done()
				err := agent.Hooks.OnStart(childCtx, agent)
				if err != nil {
					cancel()
					hooksErrors[1] = fmt.Errorf("AgentHooks.OnStart failed: %w", err)
				}
			}()
		}

		wg.Wait()
		if err := errors.Join(hooksErrors[:]...); err != nil {
			return nil, err
		}
	}

	systemPrompt, promptConfig, err := getAgentSystemPromptAndPromptConfig(ctx, agent)
	if err != nil {
		return nil, err
	}

	handoffs, err := r.getHandoffs(ctx, agent)
	if err != nil {
		return nil, err
	}

	input := ItemHelpers().InputToNewInputList(originalInput)
	if serverConversationTracker != nil {
		input = serverConversationTracker.PrepareInput(originalInput, generatedItems)
	} else {
		for _, generatedItem := range generatedItems {
			input = append(input, generatedItem.ToInputItem())
		}
	}

	newResponse, err := r.getNewResponse(
		ctx,
		agent,
		systemPrompt,
		input,
		agent.OutputType,
		allTools,
		handoffs,
		runConfig,
		toolUseTracker,
		serverConversationTracker,
		previousResponseID,
		promptConfig,
	)
	if err != nil {
		return nil, err
	}

	return r.getSingleStepResultFromResponse(
		ctx,
		agent,
		allTools,
		originalInput,
		generatedItems,
		*newResponse,
		agent.OutputType,
		handoffs,
		hooks,
		runConfig,
		toolUseTracker,
		contextWrapper,
	)
}

func getAgentSystemPromptAndPromptConfig(
	ctx context.Context,
	agent *Agent,
) (
	systemPrompt param.Opt[string],
	promptConfig responses.ResponsePromptParam,
	err error,
) {
	var promptErrors [2]error

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		systemPrompt, promptErrors[0] = agent.GetSystemPrompt(ctx)
		if promptErrors[0] != nil {
			cancel()
		}
	}()

	go func() {
		defer wg.Done()
		promptConfig, _, promptErrors[1] = agent.GetPrompt(ctx)
		if promptErrors[1] != nil {
			cancel()
		}
	}()

	wg.Wait()
	err = errors.Join(promptErrors[:]...)
	return
}

func (Runner) getSingleStepResultFromResponse(
	ctx context.Context,
	agent *Agent,
	allTools []Tool,
	originalInput Input,
	preStepItems []RunItem,
	newResponse ModelResponse,
	outputType OutputTypeInterface,
	handoffs []Handoff,
	hooks RunHooks,
	runConfig RunConfig,
	toolUseTracker *AgentToolUseTracker,
	contextWrapper *RunContextWrapper[any],
) (*SingleStepResult, error) {
	processedResponse, err := RunImpl().ProcessModelResponse(
		ctx,
		agent,
		allTools,
		newResponse,
		handoffs,
	)
	if err != nil {
		return nil, err
	}

	toolUseTracker.AddToolUse(agent, processedResponse.ToolsUsed)

	return RunImpl().ExecuteToolsAndSideEffects(
		ctx,
		agent,
		originalInput,
		preStepItems,
		newResponse,
		*processedResponse,
		outputType,
		hooks,
		runConfig,
		contextWrapper,
	)
}

func (Runner) runInputGuardrails(
	ctx context.Context,
	agent *Agent,
	guardrails []InputGuardrail,
	input Input,
) ([]InputGuardrailResult, error) {
	if len(guardrails) == 0 {
		return nil, nil
	}

	guardrailResults := make([]InputGuardrailResult, len(guardrails))
	guardrailErrors := make([]error, len(guardrails))
	var tripwireErr atomic.Pointer[InputGuardrailTripwireTriggeredError]

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var mu sync.Mutex

	var wg sync.WaitGroup
	wg.Add(len(guardrails))

	for i, guardrail := range guardrails {
		go func() {
			defer wg.Done()

			result, err := RunImpl().RunSingleInputGuardrail(childCtx, agent, guardrail, input)
			if err != nil {
				cancel()
				guardrailErrors[i] = fmt.Errorf("failed to run input guardrail %s: %w", guardrail.Name, err)
				return
			}

			if result.Output.TripwireTriggered {
				cancel() // Cancel all guardrail tasks if a tripwire is triggered.
				err := NewInputGuardrailTripwireTriggeredError(result)
				tripwireErr.Store(&err)

				mu.Lock()
				defer mu.Unlock()
				AttachErrorToCurrentSpan(ctx, tracing.SpanError{
					Message: "Guardrail tripwire triggered",
					Data:    map[string]any{"guardrail": result.Guardrail.Name},
				})

				return
			}

			guardrailResults[i] = result
		}()
	}

	wg.Wait()

	if err := tripwireErr.Load(); err != nil {
		return nil, *err
	}
	if err := errors.Join(guardrailErrors...); err != nil {
		return nil, err
	}

	return guardrailResults, nil
}

func (Runner) runOutputGuardrails(
	ctx context.Context,
	guardrails []OutputGuardrail,
	agent *Agent,
	agentOutput any,
) ([]OutputGuardrailResult, error) {
	if len(guardrails) == 0 {
		return nil, nil
	}

	guardrailResults := make([]OutputGuardrailResult, len(guardrails))
	guardrailErrors := make([]error, len(guardrails))
	var tripwireErr atomic.Pointer[OutputGuardrailTripwireTriggeredError]

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var mu sync.Mutex

	var wg sync.WaitGroup
	wg.Add(len(guardrails))

	for i, guardrail := range guardrails {
		go func() {
			defer wg.Done()

			result, err := RunImpl().RunSingleOutputGuardrail(childCtx, guardrail, agent, agentOutput)
			if err != nil {
				cancel()
				guardrailErrors[i] = fmt.Errorf("failed to run output guardrail %s: %w", guardrail.Name, err)
				return
			}

			if result.Output.TripwireTriggered {
				cancel() // Cancel all guardrail tasks if a tripwire is triggered.
				err := NewOutputGuardrailTripwireTriggeredError(result)
				tripwireErr.Store(&err)

				mu.Lock()
				defer mu.Unlock()
				AttachErrorToCurrentSpan(ctx, tracing.SpanError{
					Message: "Guardrail tripwire triggered",
					Data:    map[string]any{"guardrail": result.Guardrail.Name},
				})

				return
			}

			guardrailResults[i] = result
		}()
	}

	wg.Wait()

	if err := tripwireErr.Load(); err != nil {
		return nil, *err
	}
	if err := errors.Join(guardrailErrors...); err != nil {
		return nil, err
	}

	return guardrailResults, nil
}

func (r Runner) getNewResponse(
	ctx context.Context,
	agent *Agent,
	systemPrompt param.Opt[string],
	input []TResponseInputItem,
	outputType OutputTypeInterface,
	allTools []Tool,
	handoffs []Handoff,
	runConfig RunConfig,
	toolUseTracker *AgentToolUseTracker,
	serverConversationTracker *OpenAIServerConversationTracker,
	previousResponseID string,
	promptConfig responses.ResponsePromptParam,
) (*ModelResponse, error) {
	// Allow user to modify model input right before the call, if configured
	filtered, err := r.maybeFilterModelInput(
		ctx,
		agent,
		runConfig,
		input,
		systemPrompt,
	)
	if err != nil {
		return nil, err
	}
	if serverConversationTracker != nil {
		serverConversationTracker.MarkInputAsSent(filtered.Input)
	}

	conversationID := runConfig.ConversationID
	if serverConversationTracker != nil {
		conversationID = serverConversationTracker.ConversationID
		previousResponseID = serverConversationTracker.PreviousResponseID
	}

	model, err := r.getModel(agent, runConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to get model: %w", err)
	}

	modelSettings := agent.ModelSettings.Resolve(runConfig.ModelSettings)
	modelSettings = RunImpl().MaybeResetToolChoice(agent, toolUseTracker, modelSettings)

	// If the agent has hooks, we need to call them before and after the LLM call
	if agent.Hooks != nil {
		err = agent.Hooks.OnLLMStart(ctx, agent, filtered.Instructions, filtered.Input)
		if err != nil {
			return nil, err
		}
	}

	newResponse, err := model.GetResponse(ctx, ModelResponseParams{
		SystemInstructions: filtered.Instructions,
		Input:              InputItems(filtered.Input),
		ModelSettings:      modelSettings,
		Tools:              allTools,
		OutputType:         outputType,
		Handoffs:           handoffs,
		Tracing: GetModelTracingImpl(
			runConfig.TracingDisabled,
			runConfig.TraceIncludeSensitiveData.Or(true),
		),
		PreviousResponseID: previousResponseID,
		ConversationID:     conversationID,
		Prompt:             promptConfig,
	})
	if err != nil {
		return nil, err
	}

	// If the agent has hooks, we need to call them after the LLM call
	if agent.Hooks != nil {
		err = agent.Hooks.OnLLMEnd(ctx, agent, *newResponse)
		if err != nil {
			return nil, err
		}
	}

	if newResponse.Usage == nil {
		newResponse.Usage = &usage.Usage{Requests: 1}
	} else {
		newResponse.Usage.Requests++
	}

	if contextUsage, _ := usage.FromContext(ctx); contextUsage != nil {
		contextUsage.Add(newResponse.Usage)
	}

	if serverConversationTracker != nil {
		serverConversationTracker.TrackServerItems(newResponse)
	}

	return newResponse, err
}

func (Runner) getHandoffs(ctx context.Context, agent *Agent) ([]Handoff, error) {
	handoffs := make([]Handoff, 0, len(agent.Handoffs)+len(agent.AgentHandoffs))
	for _, h := range agent.Handoffs {
		handoffs = append(handoffs, h)
	}
	for _, a := range agent.AgentHandoffs {
		h, err := SafeHandoffFromAgent(HandoffFromAgentParams{Agent: a})
		if err != nil {
			return nil, fmt.Errorf("failed to make Handoff from Agent %q: %w", a.Name, err)
		}
		handoffs = append(handoffs, *h)
	}

	isEnabledResults := make([]bool, len(handoffs))
	isEnabledErrors := make([]error, len(handoffs))

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(len(handoffs))

	for i, handoff := range handoffs {
		go func() {
			defer wg.Done()

			if handoff.IsEnabled == nil {
				isEnabledResults[i] = true
				return
			}

			isEnabledResults[i], isEnabledErrors[i] = handoff.IsEnabled.IsEnabled(childCtx, agent)
			if isEnabledErrors[i] != nil {
				cancel()
			}
		}()
	}

	wg.Wait()
	if err := errors.Join(isEnabledErrors...); err != nil {
		return nil, err
	}

	var enabledHandoffs []Handoff
	for i, handoff := range handoffs {
		if isEnabledResults[i] {
			enabledHandoffs = append(enabledHandoffs, handoff)
		}
	}

	return enabledHandoffs, nil
}

func (Runner) getAllTools(ctx context.Context, agent *Agent) ([]Tool, error) {
	return agent.GetAllTools(ctx)
}

func (r Runner) getModel(agent *Agent, runConfig RunConfig) (Model, error) {
	modelProvider := runConfig.ModelProvider
	if modelProvider == nil {
		modelProvider = NewMultiProvider(NewMultiProviderParams{})
	}

	if runConfig.Model.Valid() {
		runConfigModel := runConfig.Model.Value
		if v, ok := runConfigModel.SafeModel(); ok {
			return v, nil
		}
		return modelProvider.GetModel(runConfigModel.ModelName())
	}

	if agent.Model.Valid() {
		agentModel := agent.Model.Value
		if v, ok := agentModel.SafeModel(); ok {
			return v, nil
		}
		return modelProvider.GetModel(agentModel.ModelName())
	}

	return modelProvider.GetModel(GetDefaultModel())
}

// prepareInputWithSession prepares input by combining it with session history if enabled.
func (r Runner) prepareInputWithSession(ctx context.Context, input Input, allowInputItems bool) (Input, error) {
	session := r.Config.Session
	if session == nil {
		return input, nil
	}

	// Validate that we don't have both a session and a list input, as this creates
	// ambiguity about whether the list should append to or replace existing session history
	if _, ok := input.(InputItems); ok {
		if !allowInputItems {
			return nil, NewUserError(
				"Cannot provide both a session and a list of input items. " +
					"When using session memory, provide only a string input to append to the " +
					"conversation, or use Session: nil and provide a list to manually manage " +
					"conversation history.",
			)
		}
		// When resuming with a session, assume the input already includes history.
		return input, nil
	}

	limit := r.Config.LimitMemory
	// Get previous conversation history
	history, err := session.GetItems(ctx, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to get session items: %w", err)
	}

	// Convert input to list format
	newInputList := ItemHelpers().InputToNewInputList(input)

	// Combine history with new input
	combinedInput := slices.Concat(history, newInputList)

	return InputItems(combinedInput), nil
}

// saveResultToSession saves the conversation turn to session.
func (r Runner) saveResultToSession(ctx context.Context, originalInput Input, result *RunResult, resumeState *RunState) error {
	session := r.Config.Session
	if session == nil {
		return nil
	}

	alreadyPersisted := 0
	if resumeState != nil {
		alreadyPersisted = int(resumeState.CurrentTurnPersistedItemCount)
	}

	newRunItems := result.NewItems
	if alreadyPersisted >= len(newRunItems) {
		newRunItems = nil
	} else if alreadyPersisted > 0 {
		newRunItems = newRunItems[alreadyPersisted:]
	}

	// Convert original input to list format if needed.
	// For resumed runs, assume prior input items are already persisted in the session.
	inputList := ItemHelpers().InputToNewInputList(originalInput)
	if resumeState != nil {
		inputList = nil
	}

	// Convert new items to input format.
	newItemsAsInput := make([]TResponseInputItem, len(newRunItems))
	for i, item := range newRunItems {
		newItemsAsInput[i] = item.ToInputItem()
	}

	// Save all items from this turn, deduplicating within the batch.
	itemsToSave := deduplicateInputItemsPreferringLatest(slices.Concat(inputList, newItemsAsInput))
	savedRunItemsCount := countSavedInputItems(newItemsAsInput, itemsToSave)

	if len(itemsToSave) > 0 {
		if err := session.AddItems(ctx, itemsToSave); err != nil {
			return fmt.Errorf("failed to add session items: %w", err)
		}
	}

	if resumeState != nil {
		resumeState.CurrentTurnPersistedItemCount = uint64(alreadyPersisted + savedRunItemsCount)
	}

	if usageTrackingSession, ok := session.(memory.RunUsageTrackingAwareSession); ok {
		var runUsage *usage.Usage
		if result != nil {
			acc := usage.NewUsage()
			hasUsage := false
			for _, modelResponse := range result.RawResponses {
				if modelResponse.Usage == nil {
					continue
				}
				acc.Add(modelResponse.Usage)
				hasUsage = true
			}
			if hasUsage {
				runUsage = acc
			}
		}
		if err := usageTrackingSession.StoreRunUsage(ctx, runUsage); err != nil {
			return fmt.Errorf("failed to store run usage: %w", err)
		}
	}

	responseID := result.LastResponseID()
	if responseID != "" {
		if compactionSession, ok := session.(memory.OpenAIResponsesCompactionAwareSession); ok {
			err := compactionSession.RunCompaction(ctx, &memory.OpenAIResponsesCompactionArgs{
				ResponseID: responseID,
			})
			if err != nil {
				return fmt.Errorf("failed to run session compaction: %w", err)
			}
		}
	}

	return nil
}

func deduplicateInputItemsPreferringLatest(items []TResponseInputItem) []TResponseInputItem {
	if len(items) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(items))
	out := make([]TResponseInputItem, 0, len(items))
	for i := len(items) - 1; i >= 0; i-- {
		key := fingerprintInputItem(items[i])
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, items[i])
	}
	slices.Reverse(out)
	return out
}

func countSavedInputItems(newItems []TResponseInputItem, itemsToSave []TResponseInputItem) int {
	if len(newItems) == 0 || len(itemsToSave) == 0 {
		return 0
	}
	counts := make(map[string]int, len(itemsToSave))
	for _, item := range itemsToSave {
		key := fingerprintInputItem(item)
		counts[key]++
	}
	saved := 0
	for _, item := range newItems {
		key := fingerprintInputItem(item)
		if counts[key] > 0 {
			counts[key]--
			saved++
		}
	}
	return saved
}

func fingerprintInputItem(item TResponseInputItem) string {
	data, err := json.Marshal(item)
	if err != nil {
		return fmt.Sprintf("%#v", item)
	}
	return string(data)
}
