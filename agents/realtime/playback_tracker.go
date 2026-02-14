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
	"strings"
	"time"
)

const (
	pcm16SampleRateHz      = 24000.0
	pcm16SampleWidthBytes  = 2.0
	g711SampleRateHz       = 8000.0
	millisecondsMultiplier = 1000.0
)

type modelAudioStateKey struct {
	itemID           string
	itemContentIndex int
}

type playbackItemKey struct {
	itemID           string
	itemContentIndex int
}

// AudioItemRef identifies an audio item and content index.
type AudioItemRef struct {
	ItemID           string
	ItemContentIndex int
}

// ModelAudioState stores timing information about a model audio item.
type ModelAudioState struct {
	InitialReceivedTime time.Time
	AudioLengthMS       float64
}

// RealtimePlaybackState describes current playback progress.
type RealtimePlaybackState struct {
	CurrentItemID           *string
	CurrentItemContentIndex *int
	ElapsedMS               *float64
}

// ModelAudioTracker tracks assistant audio delta timing as they are received.
type ModelAudioTracker struct {
	format        string
	states        map[modelAudioStateKey]ModelAudioState
	lastAudioItem *playbackItemKey
}

// NewModelAudioTracker creates an empty model audio tracker.
func NewModelAudioTracker() *ModelAudioTracker {
	return &ModelAudioTracker{
		states: make(map[modelAudioStateKey]ModelAudioState),
	}
}

// SetAudioFormat sets the audio format used for length calculations.
func (t *ModelAudioTracker) SetAudioFormat(format string) {
	t.format = format
}

// OnAudioDelta records a received audio chunk for an item.
func (t *ModelAudioTracker) OnAudioDelta(itemID string, itemContentIndex int, audioBytes []byte) {
	ms := CalculateAudioLengthMS(t.format, audioBytes)
	key := modelAudioStateKey{
		itemID:           itemID,
		itemContentIndex: itemContentIndex,
	}

	t.lastAudioItem = &playbackItemKey{
		itemID:           itemID,
		itemContentIndex: itemContentIndex,
	}

	if state, exists := t.states[key]; exists {
		state.AudioLengthMS += ms
		t.states[key] = state
		return
	}

	t.states[key] = ModelAudioState{
		InitialReceivedTime: time.Now(),
		AudioLengthMS:       ms,
	}
}

// OnInterrupted clears currently playing item tracking.
func (t *ModelAudioTracker) OnInterrupted() {
	t.lastAudioItem = nil
}

// GetState returns the current state for an item/content pair.
func (t *ModelAudioTracker) GetState(itemID string, itemContentIndex int) *ModelAudioState {
	key := modelAudioStateKey{
		itemID:           itemID,
		itemContentIndex: itemContentIndex,
	}
	state, exists := t.states[key]
	if !exists {
		return nil
	}

	result := state
	return &result
}

// GetLastAudioItem returns the most recent audio item reference.
func (t *ModelAudioTracker) GetLastAudioItem() *AudioItemRef {
	if t.lastAudioItem == nil {
		return nil
	}

	return &AudioItemRef{
		ItemID:           t.lastAudioItem.itemID,
		ItemContentIndex: t.lastAudioItem.itemContentIndex,
	}
}

// RealtimePlaybackTracker tracks actually-played audio progress for interruptions.
type RealtimePlaybackTracker struct {
	format      string
	currentItem *playbackItemKey
	elapsedMS   *float64
}

// NewRealtimePlaybackTracker creates a playback tracker.
func NewRealtimePlaybackTracker() *RealtimePlaybackTracker {
	return &RealtimePlaybackTracker{}
}

// OnPlayBytes records playback progress in bytes.
func (t *RealtimePlaybackTracker) OnPlayBytes(itemID string, itemContentIndex int, played []byte) {
	ms := CalculateAudioLengthMS(t.format, played)
	t.OnPlayMS(itemID, itemContentIndex, ms)
}

// OnPlayMS records playback progress in milliseconds.
func (t *RealtimePlaybackTracker) OnPlayMS(itemID string, itemContentIndex int, ms float64) {
	if t.currentItem == nil || t.currentItem.itemID != itemID ||
		t.currentItem.itemContentIndex != itemContentIndex {
		t.currentItem = &playbackItemKey{
			itemID:           itemID,
			itemContentIndex: itemContentIndex,
		}
		t.elapsedMS = &ms
		return
	}

	if t.elapsedMS == nil {
		t.elapsedMS = &ms
		return
	}

	updated := *t.elapsedMS + ms
	t.elapsedMS = &updated
}

// OnInterrupted resets playback state.
func (t *RealtimePlaybackTracker) OnInterrupted() {
	t.currentItem = nil
	t.elapsedMS = nil
}

// SetAudioFormat sets the audio format used when converting played bytes to ms.
func (t *RealtimePlaybackTracker) SetAudioFormat(format string) {
	t.format = format
}

// GetState returns the current playback state.
func (t *RealtimePlaybackTracker) GetState() RealtimePlaybackState {
	if t.currentItem == nil || t.elapsedMS == nil {
		return RealtimePlaybackState{}
	}

	itemID := t.currentItem.itemID
	contentIndex := t.currentItem.itemContentIndex
	elapsedMS := *t.elapsedMS

	return RealtimePlaybackState{
		CurrentItemID:           &itemID,
		CurrentItemContentIndex: &contentIndex,
		ElapsedMS:               &elapsedMS,
	}
}

// CalculateAudioLengthMS estimates audio duration in milliseconds from raw bytes.
func CalculateAudioLengthMS(format any, audioBytes []byte) float64 {
	if len(audioBytes) == 0 {
		return 0
	}

	normalizedFormat := ""
	switch v := format.(type) {
	case string:
		normalizedFormat = strings.ToLower(v)
	case *string:
		if v != nil {
			normalizedFormat = strings.ToLower(*v)
		}
	}

	if strings.HasPrefix(normalizedFormat, "g711") {
		return (float64(len(audioBytes)) / g711SampleRateHz) * millisecondsMultiplier
	}

	samples := float64(len(audioBytes)) / pcm16SampleWidthBytes
	return (samples / pcm16SampleRateHz) * millisecondsMultiplier
}
