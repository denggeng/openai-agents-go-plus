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

package tracing

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func resetTraceGlobalsForTest() {
	globalTraceProvider.Store(nil)
	globalExporter.Store(nil)
	globalProcessor.Store(nil)
	defaultProviderOnce = sync.Once{}
	defaultExporterOnce = sync.Once{}
	defaultProcessorOnce = sync.Once{}
	shutdownOnce = sync.Once{}
	shutdownHandlerRegistered.Store(false)
}

func TestTracingHasNoImportSideEffects(t *testing.T) {
	resetTraceGlobalsForTest()

	assert.Nil(t, globalTraceProvider.Load())
	assert.Nil(t, globalExporter.Load())
	assert.Nil(t, globalProcessor.Load())
	assert.False(t, shutdownHandlerRegistered.Load())
}

func TestGetTraceProviderLazilyInitializesDefaults(t *testing.T) {
	resetTraceGlobalsForTest()

	provider := GetTraceProvider()
	require.NotNil(t, provider)
	assert.NotNil(t, globalTraceProvider.Load())
	assert.NotNil(t, globalExporter.Load())
	assert.NotNil(t, globalProcessor.Load())
	assert.True(t, shutdownHandlerRegistered.Load())

	provider2 := GetTraceProvider()
	assert.Equal(t, provider, provider2)
}

func TestSetTraceProviderSkipsDefaultBootstrap(t *testing.T) {
	resetTraceGlobalsForTest()

	custom := NewDefaultTraceProvider()
	SetTraceProvider(custom)
	provider := GetTraceProvider()

	assert.Equal(t, custom, provider)
	assert.Nil(t, globalExporter.Load())
	assert.Nil(t, globalProcessor.Load())
	assert.True(t, shutdownHandlerRegistered.Load())
}
