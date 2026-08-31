# AI Image Understanding & Content Matching Engine

## 1. Problem

We want to build a backend service that automatically understands an image
library and matches relevant images to blog posts based on semantic meaning,
rather than filenames or exact keywords.

The system should analyze images using a vision model, generate structured
metadata and embeddings, and use those representations to rank images for
each blog post.

The system must also avoid incorrect recommendations. A mismatch guard will
combine image metadata, confidence scores, and semantic similarity to decide
whether a candidate is good enough to recommend. If no candidate meets the
required conditions, the system should safely reject the recommendation and
provide a human-readable explanation.

## 2. Data Model

The system will use PostgreSQL to persist images, image metadata, blog posts,
embeddings, matching suggestions, human reviews, and AI processing costs.

### Main entities

- **Images** — stores the image file information and location.
- **Image Metadata** — stores structured vision-model output such as subject,
  category, attributes, caption, confidence, and processing status.
- **Posts** — stores blog post content that needs a relevant image.
- **Embeddings** — stores vector representations used for semantic matching.
- **Suggestions** — stores ranked image recommendations, similarity scores,
  decisions, and explanations.
- **Reviews** — stores human approval or rejection of suggestions.
- **AI Costs** — records AI processing calls and their associated costs.

## 3. API Surface

The backend will expose a small REST API.

### Images

- `POST /images` — Add an image to the image library.
- `GET /images` — List images and their processing status.
- `POST /images/process` — Start background processing for unprocessed images.

### Posts

- `POST /posts` — Create a blog post.
- `GET /posts/{post_id}` — Get a blog post.

### Matching

- `GET /posts/{post_id}/images` — Return ranked image suggestions for a post,
  including similarity scores, decisions, and explanations.

### Reviews

- `GET /suggestions/{suggestion_id}` — Inspect a suggestion and its reasoning.
- `POST /suggestions/{suggestion_id}/approve` — Approve a suggestion.
- `POST /suggestions/{suggestion_id}/reject` — Reject a suggestion.

## 4. Architecture

The application will use a layered architecture that separates HTTP/API
handling, business logic, AI processing, and data persistence.

### Main layers

- **API layer** — FastAPI routes that receive requests, validate input, and
  return HTTP responses.
- **Service layer** — Contains application and business logic such as image
  processing, semantic matching, ranking, and the mismatch guard.
- **AI layer** — Handles communication with Gemini for vision analysis and
  embeddings.
- **Repository layer** — Handles database operations and persistence.
- **Background workers** — Processes image understanding and embedding
  generation asynchronously with retries.

### Main flow

Images are processed by a background job through the vision model. The
structured metadata is validated and stored, then image descriptions are
converted into embeddings.

Blog post content is also converted into an embedding. The matching service
compares the post embedding with image embeddings, ranks candidates, and
passes them through the mismatch guard.

The final result is either a ranked recommendation with an explanation or a
safe rejection when no candidate is confident enough.

## 5. Mismatch Guard

The mismatch guard is the main safety layer of the matching system. Its
purpose is to prevent incorrect image recommendations instead of always
returning the highest-ranked candidate.

The guard will consider:

- Image classification confidence.
- Semantic similarity between the post and image embeddings.
- Subject/category compatibility between the post and image metadata.

A candidate will only be accepted when it satisfies the required conditions.
Low-confidence classifications and candidates below the similarity threshold
will be rejected or flagged.

Subject mismatches will produce a clear human-readable explanation. For
example, a wolf image should be rejected for a post about red foxes even if
the wolf has a high semantic similarity score.

If no candidate passes the guard, the API will return "no confident match"
with an explanation rather than guessing.

The similarity and confidence thresholds will be tuned using the labeled
evaluation dataset rather than chosen purely by intuition.

## 6. Non-goal

This project will not build a full image management platform or public
frontend. The scope is limited to the backend AI pipeline for image
understanding, semantic matching, mismatch detection, and human review.