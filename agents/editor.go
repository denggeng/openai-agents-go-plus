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

// ApplyPatchOperationType identifies an apply_patch editor operation.
type ApplyPatchOperationType string

const (
	ApplyPatchOperationCreateFile ApplyPatchOperationType = "create_file"
	ApplyPatchOperationUpdateFile ApplyPatchOperationType = "update_file"
	ApplyPatchOperationDeleteFile ApplyPatchOperationType = "delete_file"
)

// ApplyPatchOperation represents a single apply_patch editor operation.
type ApplyPatchOperation struct {
	Type       ApplyPatchOperationType
	Path       string
	Diff       string
	CtxWrapper any
}

// ApplyPatchResultStatus defines the completion state of an apply_patch operation.
type ApplyPatchResultStatus string

const (
	ApplyPatchResultStatusCompleted ApplyPatchResultStatus = "completed"
	ApplyPatchResultStatusFailed    ApplyPatchResultStatus = "failed"
)

// ApplyPatchResult contains optional editor metadata.
type ApplyPatchResult struct {
	Status ApplyPatchResultStatus
	Output string
}

// ApplyPatchEditor applies diffs to files on disk.
type ApplyPatchEditor interface {
	CreateFile(operation ApplyPatchOperation) (any, error)
	UpdateFile(operation ApplyPatchOperation) (any, error)
	DeleteFile(operation ApplyPatchOperation) (any, error)
}
