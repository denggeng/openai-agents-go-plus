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
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewResponsesWebSocketSessionUsesWSProvider(t *testing.T) {
	client := NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{})
	session := NewResponsesWebSocketSession(ResponsesWebSocketSessionParams{
		OpenaiClient: &client,
	})
	require.NotNil(t, session)
	require.NotNil(t, session.Provider)

	model, err := session.Provider.GetModel("gpt-4.1")
	require.NoError(t, err)
	_, ok := model.(*OpenAIResponsesWSModel)
	assert.True(t, ok)

	require.NoError(t, session.Close(t.Context()))
}
