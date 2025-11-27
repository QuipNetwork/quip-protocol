// Simulate Akash provider's manifest hash computation
// This uses struct definitions similar to Akash's to test omitempty behavior
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// Simplified Akash manifest structures with omitempty tags
// Based on: https://github.com/akash-network/akash-api/blob/main/go/manifest/v2beta2/service.go

type Manifest []Group

type Group struct {
	Name     string    `json:"name"`
	Services []Service `json:"services"`
}

type Service struct {
	Name        string            `json:"name"`
	Image       string            `json:"image"`
	Command     []string          `json:"command,omitempty"`
	Args        []string          `json:"args,omitempty"`
	Env         []string          `json:"env,omitempty"`
	Resources   Resources         `json:"resources"`
	Count       uint32            `json:"count"`
	Expose      []Expose          `json:"expose,omitempty"`
	Params      *ServiceParams    `json:"params,omitempty"`
	Credentials *Credentials      `json:"credentials,omitempty"`
}

type Resources struct {
	ID        uint32          `json:"id"`
	CPU       ResourceValue   `json:"cpu"`
	Memory    ResourceValue   `json:"memory"`
	GPU       *ResourceValue  `json:"gpu,omitempty"`
	Storage   []StorageValue  `json:"storage,omitempty"`
	Endpoints []Endpoint      `json:"endpoints,omitempty"`
}

type ResourceValue struct {
	Units ResourceUnits `json:"units,omitempty"`
	Size  ResourceUnits `json:"size,omitempty"`
}

type ResourceUnits struct {
	Val string `json:"val"`
}

type StorageValue struct {
	Name string        `json:"name"`
	Size ResourceUnits `json:"size"`
}

type Endpoint struct {
	Kind           int `json:"kind,omitempty"`
	SequenceNumber int `json:"sequence_number"`
}

type Expose struct {
	Port                   uint16       `json:"port"`
	ExternalPort           uint16       `json:"externalPort"`
	Proto                  string       `json:"proto"`
	Service                string       `json:"service,omitempty"`
	Global                 bool         `json:"global"`
	Hosts                  []string     `json:"hosts,omitempty"`
	HTTPOptions            *HTTPOptions `json:"httpOptions,omitempty"`
	IP                     string       `json:"ip,omitempty"`
	EndpointSequenceNumber int          `json:"endpointSequenceNumber"`
}

type HTTPOptions struct {
	MaxBodySize uint32   `json:"maxBodySize"`
	ReadTimeout uint32   `json:"readTimeout"`
	SendTimeout uint32   `json:"sendTimeout"`
	NextTries   uint32   `json:"nextTries"`
	NextTimeout uint32   `json:"nextTimeout"`
	NextCases   []string `json:"nextCases"`
}

type ServiceParams struct {
	Storage []StorageParam `json:"storage,omitempty"`
}

type StorageParam struct {
	Name     string `json:"name"`
	Mount    string `json:"mount"`
	ReadOnly bool   `json:"readOnly"`
}

type Credentials struct {
	Host     string `json:"host"`
	Email    string `json:"email,omitempty"`
	Username string `json:"username"`
	Password string `json:"password"`
}

func main() {
	var input string

	if len(os.Args) > 1 {
		input = os.Args[1]
	} else {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading stdin: %v\n", err)
			os.Exit(1)
		}
		input = string(data)
	}

	fmt.Println("=== Akash Manifest Hash Simulation ===")
	fmt.Printf("Input length: %d bytes\n", len(input))
	fmt.Printf("Input (first 300): %s\n", truncate(input, 300))

	// Method 1: Direct hash of input
	hash1 := sha256.Sum256([]byte(input))
	fmt.Printf("\nMethod 1 - Direct SHA256:\n")
	fmt.Printf("  Hash: %s\n", hex.EncodeToString(hash1[:]))

	// Method 2: Parse into Akash structs and re-marshal (simulates provider)
	var manifest Manifest
	if err := json.Unmarshal([]byte(input), &manifest); err != nil {
		fmt.Fprintf(os.Stderr, "\nError parsing into Akash structs: %v\n", err)

		// Try parsing as generic interface for comparison
		var generic interface{}
		if err2 := json.Unmarshal([]byte(input), &generic); err2 != nil {
			fmt.Fprintf(os.Stderr, "Also failed generic parse: %v\n", err2)
		} else {
			remarshaled, _ := json.Marshal(generic)
			fmt.Printf("\nGeneric re-marshal (first 300): %s\n", truncate(string(remarshaled), 300))
		}
		os.Exit(1)
	}

	remarshaled, err := json.Marshal(manifest)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error re-marshaling: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nMethod 2 - Parse as Akash structs + re-marshal (struct field order):\n")
	fmt.Printf("  Output length: %d bytes\n", len(remarshaled))
	fmt.Printf("  Output (first 300): %s\n", truncate(string(remarshaled), 300))

	hash2 := sha256.Sum256(remarshaled)
	fmt.Printf("  Hash: %s\n", hex.EncodeToString(hash2[:]))

	// Method 3: Apply sdk.SortJSON (unmarshal to interface{} → remarshal for sorted keys)
	// This simulates what the provider does with manifest.Version()
	sorted := sortJSON(remarshaled)
	fmt.Printf("\nMethod 3 - After sdk.SortJSON (alphabetical keys):\n")
	fmt.Printf("  Output length: %d bytes\n", len(sorted))
	fmt.Printf("  Output (first 300): %s\n", truncate(string(sorted), 300))

	hash3 := sha256.Sum256(sorted)
	fmt.Printf("  Hash: %s\n", hex.EncodeToString(hash3[:]))

	// Compare
	fmt.Printf("\nComparison:\n")
	fmt.Printf("  Direct hash matches struct hash: %v\n", hash1 == hash2)
	fmt.Printf("  Direct hash matches sorted hash: %v\n", hash1 == hash3)
	fmt.Printf("  Struct hash matches sorted hash: %v\n", hash2 == hash3)
	fmt.Printf("  Input == Sorted: %v\n", input == string(sorted))

	if input != string(sorted) {
		fmt.Printf("\n  Key differences (input vs sorted):\n")
		showDiff(input, string(sorted))
	}
}

// sortJSON simulates sdk.SortJSON: unmarshal to interface{} → remarshal
// Go's json.Marshal sorts map keys alphabetically
func sortJSON(data []byte) []byte {
	var generic interface{}
	if err := json.Unmarshal(data, &generic); err != nil {
		return data
	}
	sorted, err := json.Marshal(generic)
	if err != nil {
		return data
	}
	return sorted
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func showDiff(a, b string) {
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	for i := 0; i < minLen; i++ {
		if a[i] != b[i] {
			start := i - 30
			if start < 0 {
				start = 0
			}
			end := i + 30
			if end > minLen {
				end = minLen
			}
			fmt.Printf("  Position %d:\n", i)
			fmt.Printf("    Input:  ...%s...\n", a[start:end])
			fmt.Printf("    Output: ...%s...\n", b[start:end])
			return
		}
	}

	if len(a) != len(b) {
		fmt.Printf("  Length difference: input=%d, output=%d\n", len(a), len(b))
		if len(a) > len(b) {
			fmt.Printf("  Extra in input: %q\n", a[len(b):])
		} else {
			fmt.Printf("  Extra in output: %q\n", b[len(a):])
		}
	}
}
