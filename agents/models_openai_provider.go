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
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
)

type OpenAIProviderParams struct {
	// The API key to use for the OpenAI client. If not provided, we will use the
	// default API key.
	APIKey param.Opt[string]

	// The base URL to use for the OpenAI client. If not provided, we will use the
	// default base URL.
	BaseURL param.Opt[string]

	// Optional websocket base URL used by the Responses websocket transport.
	WebsocketBaseURL param.Opt[string]

	// An optional OpenAI client to use. If not provided, we will create a new
	// OpenAI client using the APIKey and BaseURL.
	OpenaiClient *OpenaiClient

	// The organization to use for the OpenAI client.
	Organization param.Opt[string]

	// The project to use for the OpenAI client.
	Project param.Opt[string]

	// Whether to use the OpenAI responses API.
	UseResponses param.Opt[bool]

	// Whether to use websocket transport for the OpenAI responses API.
	UseResponsesWebsocket param.Opt[bool]
}

type OpenAIProvider struct {
	params             OpenAIProviderParams
	useResponses       bool
	responsesTransport OpenAIResponsesTransport
	client             *OpenaiClient
	wsModelCache       map[string]*OpenAIResponsesWSModel
	wsModelCacheMu     sync.Mutex
}

// NewOpenAIProvider creates a new OpenAI provider.
func NewOpenAIProvider(params OpenAIProviderParams) *OpenAIProvider {
	if params.OpenaiClient != nil && (params.APIKey.Valid() || params.BaseURL.Valid() || params.WebsocketBaseURL.Valid()) {
		panic(errors.New("OpenAIProvider: don't provide APIKey, BaseURL, or WebsocketBaseURL if you provide OpenaiClient"))
	}

	var useResponses bool
	if params.UseResponses.Valid() {
		useResponses = params.UseResponses.Value
	} else {
		useResponses = GetUseResponsesByDefault()
	}

	transport := GetDefaultOpenAIResponsesTransport()
	if params.UseResponsesWebsocket.Valid() {
		if params.UseResponsesWebsocket.Value {
			transport = OpenAIResponsesTransportWebsocket
		} else {
			transport = OpenAIResponsesTransportHTTP
		}
	}

	return &OpenAIProvider{
		params:             params,
		useResponses:       useResponses,
		responsesTransport: transport,
		client:             params.OpenaiClient,
		wsModelCache:       make(map[string]*OpenAIResponsesWSModel),
	}
}

func (provider *OpenAIProvider) GetModel(modelName string) (Model, error) {
	if modelName == "" {
		return nil, fmt.Errorf("cannot get OpenAI model without a name")
	}

	client := provider.getClient()

	if provider.useResponses {
		if provider.responsesTransport == OpenAIResponsesTransportWebsocket {
			return provider.getOrCreateResponsesWSModel(modelName, client), nil
		}
		return NewOpenAIResponsesModel(modelName, client), nil
	}
	return NewOpenAIChatCompletionsModel(modelName, client), nil
}

func (provider *OpenAIProvider) getOrCreateResponsesWSModel(
	modelName string,
	client OpenaiClient,
) Model {
	provider.wsModelCacheMu.Lock()
	defer provider.wsModelCacheMu.Unlock()

	if cached := provider.wsModelCache[modelName]; cached != nil {
		return cached
	}
	created := NewOpenAIResponsesWSModel(modelName, client, provider.params.WebsocketBaseURL.Or(""))
	provider.wsModelCache[modelName] = created
	return created
}

// Aclose releases provider-managed resources such as cached websocket response models.
func (provider *OpenAIProvider) Aclose(context.Context) error {
	provider.wsModelCacheMu.Lock()
	models := make([]*OpenAIResponsesWSModel, 0, len(provider.wsModelCache))
	for _, model := range provider.wsModelCache {
		if model != nil {
			models = append(models, model)
		}
	}
	clear(provider.wsModelCache)
	provider.wsModelCacheMu.Unlock()

	var closeErr error
	for _, model := range models {
		if err := model.Close(); err != nil {
			closeErr = errors.Join(closeErr, err)
		}
	}
	return closeErr
}

// We lazy load the client in case you never actually use OpenAIProvider.
// It panics if you don't have an API key set.
func (provider *OpenAIProvider) getClient() OpenaiClient {
	if provider.client == nil {
		if defaultClient := GetDefaultOpenaiClient(); defaultClient != nil {
			provider.client = defaultClient
		} else {
			var apiKey param.Opt[string]
			if provider.params.APIKey.Valid() {
				apiKey = provider.params.APIKey
			} else if defaultKey := GetDefaultOpenaiKey(); defaultKey.Valid() {
				apiKey = defaultKey
			} else if envKey := os.Getenv("OPENAI_API_KEY"); envKey != "" {
				apiKey = param.NewOpt(envKey)
			} else {
				Logger().Warn("OpenAIProvider: an API key is missing")
			}

			options := make([]option.RequestOption, 0)

			if provider.params.Organization.Valid() {
				options = append(options, option.WithOrganization(provider.params.Organization.Value))
			}
			if provider.params.Project.Valid() {
				options = append(options, option.WithProject(provider.params.Project.Value))
			}

			newClient := NewOpenaiClient(provider.params.BaseURL, apiKey, options...)
			if provider.params.WebsocketBaseURL.Valid() {
				newClient.WebsocketBaseURL = provider.params.WebsocketBaseURL
			} else if envWSBaseURL := os.Getenv("OPENAI_WEBSOCKET_BASE_URL"); envWSBaseURL != "" {
				newClient.WebsocketBaseURL = param.NewOpt(envWSBaseURL)
			}
			provider.client = &newClient
		}
	}
	return *provider.client
}
