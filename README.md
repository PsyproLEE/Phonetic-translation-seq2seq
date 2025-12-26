## Overview

The final model is a Seq2Seq architecture with attention that directly
maps Korean phonetic input to Korean semantic meaning.

Japanese sentences are used only as a source corpus to construct a new dataset,
and are not used as direct input to the final model.


## Dataset Construction

Each Japanese sentence is translated using Papago to obtain:
- its Korean pronunciation (phonetic transcription)
- its Korean semantic meaning

These pronunciation–meaning pairs are then used to train the Seq2Seq model.
This design avoids cascading errors from multi-stage translation pipelines
and enables a direct end-to-end mapping.


## Task Definition

The final task is defined as:

**Seq2Seq: Korean-pronunciation → Korean-meaning**

D = { (p_i, m_i) }

p_i: Papago-generated Korean pronunciation  
m_i: Papago-generated Korean semantic meaning

## Example

**Input**:  
`아리가또 고자이마스`

**Output**:  
`감사합니다`
