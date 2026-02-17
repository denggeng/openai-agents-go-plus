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

// ToolOutputText represents a tool output that should be sent to the model as text.
type ToolOutputText struct {
	Text string
}

// ToolOutputImage represents a tool output that should be sent to the model as an image.
// Provide either ImageURL or FileID. Optional Detail controls vision detail.
type ToolOutputImage struct {
	ImageURL string
	FileID   string
	Detail   string
}

// ToolOutputFileContent represents a tool output that should be sent to the model as a file.
// Provide one of FileData, FileURL, or FileID. Filename is optional.
type ToolOutputFileContent struct {
	FileData string
	FileURL  string
	FileID   string
	Filename string
}
