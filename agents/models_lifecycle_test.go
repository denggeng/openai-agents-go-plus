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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type closeOnlyProvider struct {
	closed bool
}

func (p *closeOnlyProvider) GetModel(string) (Model, error) { return nil, nil }
func (p *closeOnlyProvider) Aclose(context.Context) error {
	p.closed = true
	return nil
}

func TestMultiProviderAcloseCascades(t *testing.T) {
	openaiProvider := NewOpenAIProvider(OpenAIProviderParams{})
	custom := &closeOnlyProvider{}
	mp := NewMultiProvider(NewMultiProviderParams{})
	mp.OpenAIProvider = openaiProvider
	mp.ProviderMap = NewMultiProviderMap()
	mp.ProviderMap.AddProvider("custom", custom)

	require.NoError(t, mp.Aclose(t.Context()))
	assert.True(t, custom.closed)
}
