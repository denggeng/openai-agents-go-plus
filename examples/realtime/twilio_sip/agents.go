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

package main

import (
	"context"
	"strings"
	"time"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agents/extensions/handoff_prompt"
	realtime "github.com/denggeng/openai-agents-go-plus/agents/realtime"
)

const welcomeMessage = "Hello, this is ABC customer service. How can I help you today?"

// FAQLookupArgs represents FAQ lookup request inputs.
type FAQLookupArgs struct {
	Question string `json:"question"`
}

// FAQLookupTool returns a canned FAQ response for demo purposes.
func FAQLookupTool(ctx context.Context, args FAQLookupArgs) (string, error) {
	_ = ctx
	// Simulate a network lookup delay.
	time.Sleep(3 * time.Second)

	q := strings.ToLower(args.Question)
	switch {
	case strings.Contains(q, "plan"), strings.Contains(q, "wifi"), strings.Contains(q, "wi-fi"):
		return "We provide complimentary Wi-Fi. Join the ABC-Customer network.", nil
	case strings.Contains(q, "billing"), strings.Contains(q, "invoice"):
		return "Your latest invoice is available in the ABC portal under Billing > History.", nil
	case strings.Contains(q, "hours"), strings.Contains(q, "support"):
		return "Human support agents are available 24/7; transfer to the specialist if needed.", nil
	default:
		return "I'm not sure about that. Let me transfer you back to the triage agent.", nil
	}
}

// UpdateCustomerRecordArgs represents update requests for customer records.
type UpdateCustomerRecordArgs struct {
	CustomerID string `json:"customer_id"`
	Note       string `json:"note"`
}

// UpdateCustomerRecordTool records a short note about the caller.
func UpdateCustomerRecordTool(ctx context.Context, args UpdateCustomerRecordArgs) (string, error) {
	_ = ctx
	time.Sleep(1 * time.Second)
	return "Recorded note for " + args.CustomerID + ": " + args.Note, nil
}

var (
	faqLookupTool = agents.NewFunctionTool(
		"faq_lookup_tool",
		"Lookup frequently asked questions.",
		FAQLookupTool,
	)
	updateCustomerRecordTool = agents.NewFunctionTool(
		"update_customer_record",
		"Record a short note about the caller.",
		UpdateCustomerRecordTool,
	)

	faqAgent     *realtime.RealtimeAgent[any]
	recordsAgent *realtime.RealtimeAgent[any]
	triageAgent  *realtime.RealtimeAgent[any]
)

func init() {
	faqAgent = &realtime.RealtimeAgent[any]{
		Name: "FAQ Agent",
		Instructions: handoff_prompt.RecommendedPromptPrefix +
			"\nYou are an FAQ specialist. Always rely on the faq_lookup_tool for answers and keep replies " +
			"concise. If the caller needs hands-on help, transfer back to the triage agent.",
		Tools: []agents.Tool{faqLookupTool},
	}
	recordsAgent = &realtime.RealtimeAgent[any]{
		Name: "Records Agent",
		Instructions: handoff_prompt.RecommendedPromptPrefix +
			"\nYou handle structured updates. Confirm the customer's ID, capture their request in a short " +
			"note, and use the update_customer_record tool. For anything outside data updates, return to the " +
			"triage agent.",
		Tools: []agents.Tool{updateCustomerRecordTool},
	}
	triageAgent = &realtime.RealtimeAgent[any]{
		Name: "Triage Agent",
		Instructions: handoff_prompt.RecommendedPromptPrefix + " " +
			"Always begin the call by saying exactly: '" + welcomeMessage + "' before collecting details. " +
			"Once the greeting is complete, gather context and hand off to the FAQ or Records agents when appropriate.",
		Handoffs: []any{faqAgent, realtime.RealtimeHandoff(recordsAgent)},
	}

	faqAgent.Handoffs = append(faqAgent.Handoffs, triageAgent)
	recordsAgent.Handoffs = append(recordsAgent.Handoffs, triageAgent)
}

func getStartingAgent() *realtime.RealtimeAgent[any] {
	return triageAgent
}
