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
	"iter"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeTTS struct {
	strategy string
}

func (f *fakeTTS) ModelName() string { return "fake_tts" }

func (f *fakeTTS) Run(_ context.Context, text string, _ TTSModelSettings) TTSModelRunResult {
	chunk := AudioDataInt16{0, 0}.Bytes()
	var chunks [][]byte
	if f.strategy == "split_words" {
		for range strings.Fields(text) {
			chunks = append(chunks, chunk)
		}
	} else {
		chunks = append(chunks, chunk)
	}
	return &fakeTTSRunResult{chunks: chunks}
}

type fakeTTSRunResult struct {
	chunks [][]byte
	err    error
}

func (r *fakeTTSRunResult) Seq() iter.Seq[[]byte] {
	return func(yield func([]byte) bool) {
		for _, chunk := range r.chunks {
			if !yield(chunk) {
				break
			}
		}
	}
}

func (r *fakeTTSRunResult) Error() error { return r.err }

type fakeSTT struct {
	outputs []string
}

func (f *fakeSTT) ModelName() string { return "fake_stt" }

func (f *fakeSTT) pop() string {
	if len(f.outputs) == 0 {
		return ""
	}
	v := f.outputs[0]
	f.outputs = f.outputs[1:]
	return v
}

func (f *fakeSTT) Transcribe(_ context.Context, _ STTModelTranscribeParams) (string, error) {
	return f.pop(), nil
}

func (f *fakeSTT) CreateSession(_ context.Context, _ STTModelCreateSessionParams) (StreamedTranscriptionSession, error) {
	return &fakeTranscriptionSession{outputs: append([]string(nil), f.outputs...)}, nil
}

type fakeTranscriptionSession struct {
	outputs []string
}

func (s *fakeTranscriptionSession) TranscribeTurns(context.Context) StreamedTranscriptionSessionTranscribeTurns {
	return &fakeTranscriptionTurns{outputs: append([]string(nil), s.outputs...)}
}

func (s *fakeTranscriptionSession) Close(context.Context) error { return nil }

type fakeTranscriptionTurns struct {
	outputs []string
	err     error
}

func (t *fakeTranscriptionTurns) Seq() iter.Seq[string] {
	return func(yield func(string) bool) {
		for _, out := range t.outputs {
			if !yield(out) {
				break
			}
		}
	}
}

func (t *fakeTranscriptionTurns) Error() error { return t.err }

type fakeWorkflow struct {
	outputs [][]string
}

func (f *fakeWorkflow) addOutput(output []string) {
	f.outputs = append(f.outputs, output)
}

func (f *fakeWorkflow) addMultipleOutputs(outputs [][]string) {
	f.outputs = append(f.outputs, outputs...)
}

func (f *fakeWorkflow) pop() []string {
	if len(f.outputs) == 0 {
		return nil
	}
	out := f.outputs[0]
	f.outputs = f.outputs[1:]
	return out
}

func (f *fakeWorkflow) Run(context.Context, string) VoiceWorkflowBaseRunResult {
	return &fakeWorkflowRunResult{outputs: f.pop()}
}

func (f *fakeWorkflow) OnStart(context.Context) VoiceWorkflowBaseOnStartResult {
	return NoOpVoiceWorkflowBaseOnStartResult{}
}

type fakeWorkflowRunResult struct {
	outputs []string
	err     error
}

func (r *fakeWorkflowRunResult) Seq() iter.Seq[string] {
	return func(yield func(string) bool) {
		for _, out := range r.outputs {
			if !yield(out) {
				break
			}
		}
	}
}

func (r *fakeWorkflowRunResult) Error() error { return r.err }

func fakeStreamedAudioInput(count int) StreamedAudioInput {
	input := NewStreamedAudioInput()
	for i := 0; i < count; i++ {
		input.AddAudio(AudioDataInt16{0, 0})
	}
	return input
}

func collectVoiceEvents(t *testing.T, result *StreamedAudioResult) ([]string, [][]byte) {
	t.Helper()
	stream := result.Stream(context.Background())
	var events []string
	var audioChunks [][]byte
	for ev := range stream.Seq() {
		switch v := ev.(type) {
		case VoiceStreamEventAudio:
			events = append(events, "audio")
			if v.Data != nil {
				audioChunks = append(audioChunks, v.Data.Bytes())
			}
		case VoiceStreamEventLifecycle:
			events = append(events, string(v.Event))
		case VoiceStreamEventError:
			events = append(events, "error")
		}
	}
	require.NoError(t, stream.Error())
	return events, audioChunks
}

func TestVoicePipelineRunSingleTurn(t *testing.T) {
	fakeStt := &fakeSTT{outputs: []string{"first"}}
	workflow := &fakeWorkflow{outputs: [][]string{{"out_1"}}}
	fakeTts := &fakeTTS{}
	config := VoicePipelineConfig{
		TTSSettings: TTSModelSettings{BufferSize: 1},
	}
	pipeline, err := NewVoicePipeline(VoicePipelineParams{
		Workflow: workflow,
		STTModel: fakeStt,
		TTSModel: fakeTts,
		Config:   config,
	})
	require.NoError(t, err)

	audioInput := AudioInput{Buffer: AudioDataInt16{0, 0}}
	result, err := pipeline.Run(context.Background(), audioInput)
	require.NoError(t, err)

	events, audioChunks := collectVoiceEvents(t, result)
	assert.Equal(t, []string{"turn_started", "audio", "turn_ended", "session_ended"}, events)
	require.Len(t, audioChunks, 1)
	assert.Equal(t, AudioDataInt16{0, 0}.Bytes(), audioChunks[0])
}

func TestVoicePipelineStreamedAudioInput(t *testing.T) {
	fakeStt := &fakeSTT{outputs: []string{"first", "second"}}
	workflow := &fakeWorkflow{outputs: [][]string{{"out_1"}, {"out_2"}}}
	fakeTts := &fakeTTS{}

	pipeline, err := NewVoicePipeline(VoicePipelineParams{
		Workflow: workflow,
		STTModel: fakeStt,
		TTSModel: fakeTts,
	})
	require.NoError(t, err)

	result, err := pipeline.Run(context.Background(), fakeStreamedAudioInput(2))
	require.NoError(t, err)

	events, audioChunks := collectVoiceEvents(t, result)
	assert.Equal(t, []string{
		"turn_started",
		"audio",
		"turn_ended",
		"turn_started",
		"audio",
		"turn_ended",
		"session_ended",
	}, events)
	require.Len(t, audioChunks, 2)
	assert.Equal(t, AudioDataInt16{0, 0}.Bytes(), audioChunks[0])
	assert.Equal(t, AudioDataInt16{0, 0}.Bytes(), audioChunks[1])
}

func TestVoicePipelineRunSingleTurnSplitWords(t *testing.T) {
	fakeStt := &fakeSTT{outputs: []string{"first"}}
	workflow := &fakeWorkflow{outputs: [][]string{{"foo bar baz"}}}
	fakeTts := &fakeTTS{strategy: "split_words"}
	config := VoicePipelineConfig{
		TTSSettings: TTSModelSettings{BufferSize: 1},
	}
	pipeline, err := NewVoicePipeline(VoicePipelineParams{
		Workflow: workflow,
		STTModel: fakeStt,
		TTSModel: fakeTts,
		Config:   config,
	})
	require.NoError(t, err)

	audioInput := AudioInput{Buffer: AudioDataInt16{0, 0}}
	result, err := pipeline.Run(context.Background(), audioInput)
	require.NoError(t, err)

	events, audioChunks := collectVoiceEvents(t, result)
	assert.Equal(t, []string{"turn_started", "audio", "audio", "audio", "turn_ended", "session_ended"}, events)
	require.Len(t, audioChunks, 3)
	for _, chunk := range audioChunks {
		assert.Equal(t, AudioDataInt16{0, 0}.Bytes(), chunk)
	}
}

func TestVoicePipelineRunMultiTurnSplitWords(t *testing.T) {
	fakeStt := &fakeSTT{outputs: []string{"first", "second"}}
	workflow := &fakeWorkflow{outputs: [][]string{{"foo bar baz"}, {"foo2 bar2 baz2"}}}
	fakeTts := &fakeTTS{strategy: "split_words"}
	config := VoicePipelineConfig{
		TTSSettings: TTSModelSettings{BufferSize: 1},
	}
	pipeline, err := NewVoicePipeline(VoicePipelineParams{
		Workflow: workflow,
		STTModel: fakeStt,
		TTSModel: fakeTts,
		Config:   config,
	})
	require.NoError(t, err)

	result, err := pipeline.Run(context.Background(), fakeStreamedAudioInput(6))
	require.NoError(t, err)

	events, audioChunks := collectVoiceEvents(t, result)
	assert.Equal(t, []string{
		"turn_started",
		"audio",
		"audio",
		"audio",
		"turn_ended",
		"turn_started",
		"audio",
		"audio",
		"audio",
		"turn_ended",
		"session_ended",
	}, events)
	require.Len(t, audioChunks, 6)
	for _, chunk := range audioChunks {
		assert.Equal(t, AudioDataInt16{0, 0}.Bytes(), chunk)
	}
}

func TestVoicePipelineFloat32(t *testing.T) {
	fakeStt := &fakeSTT{outputs: []string{"first"}}
	workflow := &fakeWorkflow{outputs: [][]string{{"out_1"}}}
	fakeTts := &fakeTTS{}
	config := VoicePipelineConfig{
		TTSSettings: TTSModelSettings{
			BufferSize:    1,
			AudioDataType: param.NewOpt[AudioDataType](AudioDataTypeFloat32),
		},
	}
	pipeline, err := NewVoicePipeline(VoicePipelineParams{
		Workflow: workflow,
		STTModel: fakeStt,
		TTSModel: fakeTts,
		Config:   config,
	})
	require.NoError(t, err)

	audioInput := AudioInput{Buffer: AudioDataInt16{0, 0}}
	result, err := pipeline.Run(context.Background(), audioInput)
	require.NoError(t, err)

	events, audioChunks := collectVoiceEvents(t, result)
	assert.Equal(t, []string{"turn_started", "audio", "turn_ended", "session_ended"}, events)
	require.Len(t, audioChunks, 1)
	assert.Equal(t, AudioDataFloat32{0, 0}.Bytes(), audioChunks[0])
}

func TestVoicePipelineTransformData(t *testing.T) {
	fakeStt := &fakeSTT{outputs: []string{"first"}}
	workflow := &fakeWorkflow{outputs: [][]string{{"out_1"}}}
	fakeTts := &fakeTTS{}
	transform := func(_ context.Context, data AudioData) (AudioData, error) {
		return data.Int16(), nil
	}
	config := VoicePipelineConfig{
		TTSSettings: TTSModelSettings{
			BufferSize:    1,
			AudioDataType: param.NewOpt[AudioDataType](AudioDataTypeFloat32),
			TransformData: transform,
		},
	}
	pipeline, err := NewVoicePipeline(VoicePipelineParams{
		Workflow: workflow,
		STTModel: fakeStt,
		TTSModel: fakeTts,
		Config:   config,
	})
	require.NoError(t, err)

	audioInput := AudioInput{Buffer: AudioDataInt16{0, 0}}
	result, err := pipeline.Run(context.Background(), audioInput)
	require.NoError(t, err)

	events, audioChunks := collectVoiceEvents(t, result)
	assert.Equal(t, []string{"turn_started", "audio", "turn_ended", "session_ended"}, events)
	require.Len(t, audioChunks, 1)
	assert.Equal(t, AudioDataInt16{0, 0}.Bytes(), audioChunks[0])
}
