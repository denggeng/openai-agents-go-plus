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

package realtime

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRealtimePlaybackTrackerOnPlayBytesAndState(t *testing.T) {
	tracker := NewRealtimePlaybackTracker()
	tracker.SetAudioFormat("pcm16")

	tracker.OnPlayBytes("item1", 0, make([]byte, 48000))
	state := tracker.GetState()
	require.NotNil(t, state.CurrentItemID)
	require.NotNil(t, state.CurrentItemContentIndex)
	require.NotNil(t, state.ElapsedMS)
	assert.Equal(t, "item1", *state.CurrentItemID)
	assert.Equal(t, 0, *state.CurrentItemContentIndex)
	assert.InDelta(t, 1000.0, *state.ElapsedMS, 1e-6)

	tracker.OnPlayMS("item1", 0, 500.0)
	state2 := tracker.GetState()
	require.NotNil(t, state2.ElapsedMS)
	assert.InDelta(t, 1500.0, *state2.ElapsedMS, 1e-6)

	tracker.OnInterrupted()
	state3 := tracker.GetState()
	assert.Nil(t, state3.CurrentItemID)
	assert.Nil(t, state3.CurrentItemContentIndex)
	assert.Nil(t, state3.ElapsedMS)
}

func TestModelAudioTrackerStateAccumulationAcrossDeltas(t *testing.T) {
	tracker := NewModelAudioTracker()
	tracker.SetAudioFormat("pcm16")

	tracker.OnAudioDelta("item_1", 0, []byte("test"))
	tracker.OnAudioDelta("item_1", 0, []byte("more"))

	state := tracker.GetState("item_1", 0)
	require.NotNil(t, state)
	expectedLength := (8.0 / (24000.0 * 2.0)) * 1000.0
	assert.InDelta(t, expectedLength, state.AudioLengthMS, 1e-6)

	last := tracker.GetLastAudioItem()
	require.NotNil(t, last)
	assert.Equal(t, "item_1", last.ItemID)
	assert.Equal(t, 0, last.ItemContentIndex)
}

func TestTrackerStateCleanupOnInterruption(t *testing.T) {
	modelTracker := NewModelAudioTracker()
	modelTracker.SetAudioFormat("pcm16")
	modelTracker.OnAudioDelta("item_1", 0, []byte("test"))
	require.NotNil(t, modelTracker.GetLastAudioItem())

	modelTracker.OnInterrupted()
	assert.Nil(t, modelTracker.GetLastAudioItem())

	playbackTracker := NewRealtimePlaybackTracker()
	playbackTracker.OnPlayMS("item_1", 0, 100.0)
	beforeInterrupt := playbackTracker.GetState()
	require.NotNil(t, beforeInterrupt.CurrentItemID)
	require.NotNil(t, beforeInterrupt.ElapsedMS)
	assert.Equal(t, "item_1", *beforeInterrupt.CurrentItemID)
	assert.Equal(t, 100.0, *beforeInterrupt.ElapsedMS)

	playbackTracker.OnInterrupted()
	afterInterrupt := playbackTracker.GetState()
	assert.Nil(t, afterInterrupt.CurrentItemID)
	assert.Nil(t, afterInterrupt.ElapsedMS)
}

func TestCalculateAudioLengthMSWithDifferentFormats(t *testing.T) {
	g711Length := CalculateAudioLengthMS("g711_ulaw", []byte("12345678"))
	assert.InDelta(t, 1.0, g711Length, 1e-6)

	pcmBytes := []byte("test")
	pcmLength := CalculateAudioLengthMS("pcm16", pcmBytes)
	expectedPCM := (float64(len(pcmBytes)) / (24000.0 * 2.0)) * 1000.0
	assert.InDelta(t, expectedPCM, pcmLength, 1e-6)

	noneLength := CalculateAudioLengthMS(nil, pcmBytes)
	assert.InDelta(t, expectedPCM, noneLength, 1e-6)
}
