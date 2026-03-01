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
	"fmt"
	"sync/atomic"

	"github.com/denggeng/openai-agents-go-plus/tracing"
	"github.com/openai/openai-go/v3/packages/param"
)

var (
	defaultOpenaiKey          atomic.Pointer[string]
	defaultOpenaiClient       atomic.Pointer[OpenaiClient]
	useResponsesByDefault     atomic.Bool
	defaultResponsesTransport atomic.Pointer[OpenAIResponsesTransport]
)

func init() {
	useResponsesByDefault.Store(true)
	defaultTransport := OpenAIResponsesTransportHTTP
	defaultResponsesTransport.Store(&defaultTransport)
}

// SetDefaultOpenaiKey sets the default OpenAI API key to use for LLM requests (and optionally tracing).
// This is only necessary if the OPENAI_API_KEY environment variable is not already set.
//
// If provided, this key will be used instead of the OPENAI_API_KEY environment variable.
//
// useForTracing indicates whether to also use this key to send traces to OpenAI.
// If false, you'll either need to set the OPENAI_API_KEY environment variable or call
// tracing.SetTracingExportAPIKey with the API key you want to use for tracing.
func SetDefaultOpenaiKey(key string, useForTracing bool) {
	defaultOpenaiKey.Store(&key)

	if useForTracing {
		tracing.SetTracingExportAPIKey(key)
	}
}

func GetDefaultOpenaiKey() param.Opt[string] {
	v := defaultOpenaiKey.Load()
	if v == nil {
		return param.Opt[string]{}
	}
	return param.NewOpt(*v)
}

// SetDefaultOpenaiClient sets the default OpenAI client to use for LLM requests and/or tracing.
// If provided, this client will be used instead of the default OpenAI client.
//
// useForTracing indicates whether to use the API key from this client for uploading traces.
// If false, you'll either need to set the OPENAI_API_KEY environment variable or call
// tracing.SetTracingExportAPIKey with the API key you want to use for tracing.
func SetDefaultOpenaiClient(client OpenaiClient, useForTracing bool) {
	defaultOpenaiClient.Store(&client)

	if useForTracing && client.APIKey.Valid() {
		tracing.SetTracingExportAPIKey(client.APIKey.Value)
	}
}

func GetDefaultOpenaiClient() *OpenaiClient {
	return defaultOpenaiClient.Load()
}

func SetUseResponsesByDefault(useResponses bool) {
	useResponsesByDefault.Store(useResponses)
}

func GetUseResponsesByDefault() bool {
	return useResponsesByDefault.Load()
}

// OpenAIResponsesTransport controls which transport is used for Responses API calls.
type OpenAIResponsesTransport string

const (
	OpenAIResponsesTransportHTTP      OpenAIResponsesTransport = "http"
	OpenAIResponsesTransportWebsocket OpenAIResponsesTransport = "websocket"
)

func SetDefaultOpenAIResponsesTransport(transport OpenAIResponsesTransport) {
	setDefaultResponsesTransport(transport)
}

func GetDefaultOpenAIResponsesTransport() OpenAIResponsesTransport {
	return getDefaultResponsesTransport()
}

// SetDefaultOpenaiResponsesTransport is the backwards-compatible alias.
func SetDefaultOpenaiResponsesTransport(transport OpenAIResponsesTransport) {
	setDefaultResponsesTransport(transport)
}

// GetDefaultOpenaiResponsesTransport is the backwards-compatible alias.
func GetDefaultOpenaiResponsesTransport() OpenAIResponsesTransport {
	return getDefaultResponsesTransport()
}

func setDefaultResponsesTransport(transport OpenAIResponsesTransport) {
	switch transport {
	case OpenAIResponsesTransportHTTP, OpenAIResponsesTransportWebsocket:
		transportCopy := transport
		defaultResponsesTransport.Store(&transportCopy)
	default:
		panic(fmt.Errorf("invalid OpenAIResponsesTransport value %q", transport))
	}
}

func getDefaultResponsesTransport() OpenAIResponsesTransport {
	if transport := defaultResponsesTransport.Load(); transport != nil {
		return *transport
	}
	return OpenAIResponsesTransportHTTP
}

func SetUseResponsesWebsocketByDefault(useResponsesWebsocket bool) {
	if useResponsesWebsocket {
		SetDefaultOpenAIResponsesTransport(OpenAIResponsesTransportWebsocket)
		return
	}
	SetDefaultOpenAIResponsesTransport(OpenAIResponsesTransportHTTP)
}

func GetUseResponsesWebsocketByDefault() bool {
	return GetDefaultOpenAIResponsesTransport() == OpenAIResponsesTransportWebsocket
}

// SetDefaultOpenaiAPI set the default API to use for OpenAI LLM requests.
// By default, we will use the responses API, but you can set this to use the
// chat completions API instead.
func SetDefaultOpenaiAPI(api OpenaiAPIType) {
	switch api {
	case OpenaiAPITypeChatCompletions:
		SetUseResponsesByDefault(false)
	case OpenaiAPITypeResponses:
		SetUseResponsesByDefault(true)
	default:
		panic(fmt.Errorf("invalid OpenaiAPIType value %q", api))
	}
}

type OpenaiAPIType string

const (
	OpenaiAPITypeChatCompletions OpenaiAPIType = "chat_completions"
	OpenaiAPITypeResponses       OpenaiAPIType = "responses"
)

func ClearOpenaiSettings() {
	defaultOpenaiKey.Store(nil)
	defaultOpenaiClient.Store(nil)
	useResponsesByDefault.Store(true)
	defaultTransport := OpenAIResponsesTransportHTTP
	defaultResponsesTransport.Store(&defaultTransport)
}
