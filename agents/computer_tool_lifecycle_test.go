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

package agents_test

import (
	"context"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/denggeng/openai-agents-go-plus/computer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type lifecycleComputer struct {
	label string
}

func (c *lifecycleComputer) Environment(context.Context) (computer.Environment, error) {
	return computer.EnvironmentMac, nil
}

func (c *lifecycleComputer) Dimensions(context.Context) (computer.Dimensions, error) {
	return computer.Dimensions{Width: 1, Height: 1}, nil
}

func (c *lifecycleComputer) Screenshot(context.Context) (string, error) { return "img", nil }
func (c *lifecycleComputer) Click(context.Context, int64, int64, computer.Button) error {
	return nil
}
func (c *lifecycleComputer) DoubleClick(context.Context, int64, int64) error { return nil }
func (c *lifecycleComputer) Scroll(context.Context, int64, int64, int64, int64) error {
	return nil
}
func (c *lifecycleComputer) Type(context.Context, string) error              { return nil }
func (c *lifecycleComputer) Wait(context.Context) error                      { return nil }
func (c *lifecycleComputer) Move(context.Context, int64, int64) error        { return nil }
func (c *lifecycleComputer) Keypress(context.Context, []string) error        { return nil }
func (c *lifecycleComputer) Drag(context.Context, []computer.Position) error { return nil }

type methodValueComputerFactoryOwner struct {
	label string
}

func (o *methodValueComputerFactoryOwner) CreateComputer(
	context.Context,
	*agents.RunContextWrapper[any],
) (computer.Computer, error) {
	return &lifecycleComputer{label: o.label}, nil
}

func TestResolveComputerPerRunContext(t *testing.T) {
	counter := 0
	tool := agents.ComputerTool{
		ComputerFactory: func(context.Context, *agents.RunContextWrapper[any]) (computer.Computer, error) {
			counter++
			return &lifecycleComputer{label: "computer"}, nil
		},
	}

	ctxA := agents.NewRunContextWrapper[any](nil)
	ctxB := agents.NewRunContextWrapper[any](nil)

	compA1, err := agents.ResolveComputer(t.Context(), &tool, ctxA)
	require.NoError(t, err)
	compA2, err := agents.ResolveComputer(t.Context(), &tool, ctxA)
	require.NoError(t, err)
	compB1, err := agents.ResolveComputer(t.Context(), &tool, ctxB)
	require.NoError(t, err)

	assert.Same(t, compA1, compA2)
	assert.NotSame(t, compA1, compB1)
	assert.Equal(t, 2, counter)

	require.NoError(t, agents.DisposeResolvedComputers(t.Context(), ctxA))
	compA3, err := agents.ResolveComputer(t.Context(), &tool, ctxA)
	require.NoError(t, err)
	assert.NotSame(t, compA1, compA3)
	assert.Equal(t, 3, counter)

	require.NoError(t, agents.DisposeResolvedComputers(t.Context(), ctxB))
	require.NoError(t, agents.DisposeResolvedComputers(t.Context(), ctxA))
}

func TestRunnerDisposesComputerAfterRun(t *testing.T) {
	var (
		createCount  int
		disposeCount int
		createdComp  computer.Computer
		disposedComp computer.Computer
		createCtx    *agents.RunContextWrapper[any]
		disposeCtx   *agents.RunContextWrapper[any]
	)

	tool := agents.ComputerTool{
		ComputerProvider: &agents.ComputerProvider{
			Create: func(_ context.Context, runContext *agents.RunContextWrapper[any]) (computer.Computer, error) {
				createCount++
				createCtx = runContext
				createdComp = &lifecycleComputer{label: "created"}
				return createdComp, nil
			},
			Dispose: func(_ context.Context, runContext *agents.RunContextWrapper[any], comp computer.Computer) error {
				disposeCount++
				disposeCtx = runContext
				disposedComp = comp
				return nil
			},
		},
	}

	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("done")}},
	})
	agent := agents.New("ComputerAgent").WithModelInstance(model).WithTools(tool)

	result, err := agents.Run(t.Context(), agent, "hello")
	require.NoError(t, err)
	assert.Equal(t, "done", result.FinalOutput)
	assert.Equal(t, 1, createCount)
	assert.Equal(t, 1, disposeCount)
	assert.Same(t, createCtx, disposeCtx)
	assert.Same(t, createdComp, disposedComp)
}

func TestStreamedRunDisposesComputerAfterCompletion(t *testing.T) {
	var (
		createCount  int
		disposeCount int
		createdComp  computer.Computer
		disposedComp computer.Computer
		createCtx    *agents.RunContextWrapper[any]
		disposeCtx   *agents.RunContextWrapper[any]
	)

	tool := agents.ComputerTool{
		ComputerProvider: &agents.ComputerProvider{
			Create: func(_ context.Context, runContext *agents.RunContextWrapper[any]) (computer.Computer, error) {
				createCount++
				createCtx = runContext
				createdComp = &lifecycleComputer{label: "streaming"}
				return createdComp, nil
			},
			Dispose: func(_ context.Context, runContext *agents.RunContextWrapper[any], comp computer.Computer) error {
				disposeCount++
				disposeCtx = runContext
				disposedComp = comp
				return nil
			},
		},
	}

	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("done")}},
	})
	agent := agents.New("ComputerAgent").WithModelInstance(model).WithTools(tool)

	streamed, err := agents.RunStreamed(t.Context(), agent, "hello")
	require.NoError(t, err)
	require.NoError(t, streamed.StreamEvents(func(agents.StreamEvent) error { return nil }))
	assert.Equal(t, "done", streamed.FinalOutput())
	assert.Equal(t, 1, createCount)
	assert.Equal(t, 1, disposeCount)
	assert.Same(t, createCtx, disposeCtx)
	assert.Same(t, createdComp, disposedComp)
}

func TestResolveComputerFactoryMethodValueNoCacheCollision(t *testing.T) {
	runContext := agents.NewRunContextWrapper[any](nil)

	ownerA := &methodValueComputerFactoryOwner{label: "A"}
	ownerB := &methodValueComputerFactoryOwner{label: "B"}
	toolA := agents.ComputerTool{ComputerFactory: ownerA.CreateComputer}
	toolB := agents.ComputerTool{ComputerFactory: ownerB.CreateComputer}

	compA, err := agents.ResolveComputer(t.Context(), &toolA, runContext)
	require.NoError(t, err)
	compB, err := agents.ResolveComputer(t.Context(), &toolB, runContext)
	require.NoError(t, err)

	require.NotSame(t, compA, compB)
	compAConcrete := compA.(*lifecycleComputer)
	compBConcrete := compB.(*lifecycleComputer)
	assert.Equal(t, "A", compAConcrete.label)
	assert.Equal(t, "B", compBConcrete.label)
}
