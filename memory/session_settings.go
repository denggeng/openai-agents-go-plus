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

package memory

// SessionSettings holds optional configuration for session operations.
type SessionSettings struct {
	// Limit sets the maximum number of items to retrieve. If nil, retrieves all items.
	Limit *int
}

// Resolve overlays non-nil values from override onto the receiver.
func (s *SessionSettings) Resolve(override *SessionSettings) *SessionSettings {
	if s == nil && override == nil {
		return nil
	}

	base := SessionSettings{}
	if s != nil {
		base = *s
	}
	if override == nil {
		return &base
	}

	if override.Limit != nil {
		base.Limit = override.Limit
	}
	return &base
}

// ResolveSessionLimit returns the effective limit based on an explicit value
// and optional session settings.
func ResolveSessionLimit(explicit *int, settings *SessionSettings) *int {
	if explicit != nil {
		return explicit
	}
	if settings != nil {
		return settings.Limit
	}
	return nil
}
