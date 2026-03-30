# Prompt Presets Guide

> Complete guide to AI-rewriter prompt presets with examples and best practices.

---

## Table of Contents

- [Overview](#overview)
- [Available Presets](#available-presets)
  - [Literary Editor](#literary-editor-literary)
  - [Academic Style](#academic-style-academic)
  - [Simplified](#simplified-simplified)
  - [Creative Enhancement](#creative-enhancement-creative)
  - [Translation with Adaptation](#translation-with-adaptation-translation)
- [Choosing the Right Preset](#choosing-the-right-preset)
- [Custom Instructions](#custom-instructions)
- [Examples](#examples)

---

## Overview

AI-rewriter provides 5 built-in prompt presets, each optimized for specific text types and rewriting goals. Each preset includes:

- **System prompt** — Instructions for the LLM on how to rewrite
- **Localized name** — Display name in UI (EN/RU/ZH)
- **Category** — Grouping for UI organization
- **Tags** — Keywords for search and filtering

---

## Available Presets

### Literary Editor (`literary`)

**Best for:** Fiction, novels, short stories, literary prose

**Style:** Engaging narrative, vivid descriptions, smooth flow

**Description:** Transforms text into polished literary prose while preserving the original meaning and emotional impact. Ideal for fiction writers looking to enhance their prose style.

**Characteristics:**
- Rich, evocative language
- Strong narrative voice
- Vivid sensory details
- Smooth transitions between scenes
- Character-driven prose

**Example Use Case:**
```
Original: "She walked into the room. It was dark. She felt scared."
Rewritten: "She stepped across the threshold into a darkness so complete 
it seemed to press against her eyes. Her heart quickened, each beat 
a small rebellion against the silence that surrounded her."
```

**When to Use:**
- Novel chapters
- Short stories
- Literary essays
- Creative non-fiction

---

### Academic Style (`academic`)

**Best for:** Research papers, academic articles, dissertations, scholarly texts

**Style:** Formal, precise, scholarly, objective

**Description:** Rewrites text to meet academic writing standards with proper structure, precise terminology, and scholarly tone. Maintains objectivity while improving clarity.

**Characteristics:**
- Formal academic register
- Precise terminology
- Clear argument structure
- Objective tone
- Proper citations integration

**Example Use Case:**
```
Original: "The experiment worked pretty well and we found some 
interesting stuff about how people remember things."
Rewritten: "The experimental methodology demonstrated significant 
efficacy, yielding substantive findings regarding cognitive 
retention mechanisms in human subjects."
```

**When to Use:**
- Research papers
- Journal articles
- Thesis chapters
- Academic proposals
- Literature reviews

---

### Simplified (`simplified`)

**Best for:** Technical documentation, educational content, public communication

**Style:** Clear, simple, accessible, direct

**Description:** Makes complex content accessible to broader audiences by simplifying vocabulary, shortening sentences, and clarifying concepts without losing accuracy.

**Characteristics:**
- Simple vocabulary
- Short, clear sentences
- Active voice
- Concrete examples
- Logical flow

**Example Use Case:**
```
Original: "The implementation of quantum computational algorithms 
necessitates a fundamental reconceptualization of information 
processing paradigms."
Rewritten: "Quantum computers need us to think about computing in 
new ways. Instead of regular bits that are either 0 or 1, quantum 
computers use qubits that can be both at once."
```

**When to Use:**
- Technical documentation
- Educational materials
- Public-facing content
- Instructional guides
- Explainer articles

---

### Creative Enhancement (`creative`)

**Best for:** Marketing copy, creative writing, brand storytelling

**Style:** Vivid, evocative, dynamic, memorable

**Description:** Adds creative flair and emotional resonance to text. Perfect for making content more engaging, memorable, and impactful.

**Characteristics:**
- Dynamic, varied sentence structure
- Emotional hooks
- Memorable phrases
- Sensory language
- Creative metaphors

**Example Use Case:**
```
Original: "Our coffee is good. We roast it carefully. Try it today."
Rewritten: "Every cup tells a story of sun-drenched mountains and 
careful hands. Our master roasters coax out flavors that dance on 
your tongue—chocolate, citrus, a whisper of caramel. This isn't 
just coffee. It's an invitation to savor the extraordinary."
```

**When to Use:**
- Marketing materials
- Brand storytelling
- Product descriptions
- Creative writing exercises
- Social media content

---

### Translation with Adaptation (`translation`)

**Best for:** Cross-language rewriting, localization, cultural adaptation

**Style:** Culturally adapted, natural, fluent

**Description:** Translates and adapts text for target language audiences, considering cultural nuances, idioms, and local expressions. Goes beyond literal translation.

**Characteristics:**
- Cultural adaptation
- Natural phrasing in target language
- Idiomatic expressions
- Local references
- Tone preservation

**Example Use Case:**
```
Original (English): "It's raining cats and dogs outside."
Adapted to Russian: "На улице льёт как из ведра." 
(Pouring like from a bucket - natural Russian idiom)

Original (English): "Break a leg!"
Adapted to Chinese: "祝你好运！" 
(Good luck! - culturally appropriate equivalent)
```

**When to Use:**
- Book translations
- Content localization
- Cross-cultural communication
- International marketing
- Subtitle adaptation

---

## Choosing the Right Preset

### Decision Matrix

| Your Goal | Recommended Preset |
|-----------|-------------------|
| Polish a novel chapter | Literary Editor |
| Improve a research paper | Academic Style |
| Make technical docs accessible | Simplified |
| Enhance marketing copy | Creative Enhancement |
| Translate with cultural adaptation | Translation |

### By Content Type

| Content Type | Primary | Alternative |
|--------------|---------|-------------|
| Fiction/Novels | Literary | Creative |
| Academic Papers | Academic | Simplified |
| Technical Docs | Simplified | Academic |
| Marketing Copy | Creative | Literary |
| Translations | Translation | Literary |
| Educational Content | Simplified | Academic |
| Blog Posts | Creative | Literary |

---

## Custom Instructions

In addition to presets, you can provide custom instructions through two fields:

### Style Field

Add specific style requirements:

**Examples:**
- "Use British English spelling"
- "Maintain a conversational tone"
- "Write in first-person perspective"
- "Use short paragraphs for web readability"
- "Include humor where appropriate"

### Goal Field

Specify the rewriting objective:

**Examples:**
- "Make it 20% shorter while keeping all key points"
- "Increase emotional impact"
- "Simplify for a 12-year-old audience"
- "Add more concrete examples"
- "Strengthen the call-to-action"

---

## Examples

### Example 1: Novel Chapter

**Preset:** Literary Editor
**Style:** "Gothic atmosphere, Victorian setting"
**Goal:** "Increase tension in the opening scene"

**Before:**
> The house was old. Nobody had lived there for years. John walked up to the door.

**After:**
> The manor loomed against the bruised sky, its windows like hollow eyes that had witnessed decades of decay. No soul had crossed its threshold in a generation—until now. John's footsteps crunched on the gravel, each one bringing him closer to secrets the house had long kept buried.

---

### Example 2: Research Abstract

**Preset:** Academic Style
**Style:** "APA style, psychology journal"
**Goal:** "Clarify methodology section"

**Before:**
> We did a study with 50 people. We showed them pictures and asked questions. The results were interesting.

**After:**
> This study employed a between-subjects experimental design with 50 participants (n=25 per condition). Participants were presented with a series of visual stimuli followed by semi-structured interview questions. Results indicated statistically significant differences between conditions (p < .05), suggesting meaningful variations in participant responses.

---

### Example 3: Product Description

**Preset:** Creative Enhancement
**Style:** "Luxury brand voice, sophisticated"
**Goal:** "Emphasize exclusivity and craftsmanship"

**Before:**
> This watch is made of gold. It has a leather strap. It keeps good time.

**After:**
> Crafted from 18-karat gold and paired with hand-stitched Italian leather, this timepiece is more than an instrument—it's a legacy. Each watch requires 200 hours of meticulous assembly by master horologists whose families have practiced this art for generations. When you wear it, you carry a century of excellence on your wrist.

---

### Example 4: Technical Tutorial

**Preset:** Simplified
**Style:** "For non-technical users"
**Goal:** "Make it actionable with clear steps"

**Before:**
> The configuration file must be modified to specify the appropriate parameters for the database connection string, including but not limited to the hostname, port, authentication credentials, and database name.

**After:**
> To connect to your database, open the config file and add these details:
> 1. **Host:** The server address (like `localhost` or `192.168.1.1`)
> 2. **Port:** Usually `5432` for PostgreSQL
> 3. **Username:** Your database username
> 4. **Password:** Your database password
> 5. **Database name:** What you called your database

---

## Tips for Best Results

1. **Match preset to content type** — Using the wrong preset can produce unexpected results
2. **Use custom instructions sparingly** — Let the preset do most of the work
3. **Be specific in goals** — "Make it 30% shorter" works better than "shorten it"
4. **Review and iterate** — First results may need refinement
5. **Combine presets with style** — Use preset as base, add style for customization

---

## See Also

- [README.md](README.md) - General documentation
- [API.md](API.md) - API reference for programmatic access
