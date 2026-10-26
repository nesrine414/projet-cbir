# API

FastAPI serves the browser app and a JSON search API. Start it from the repository root with `uvicorn api.main:app --reload`; interactive schemas are at `/docs`.

## `GET /datasets`

Returns collection image counts and the descriptors present in each visual index.

```json
{
  "datasets": [
    {
      "id": "objects",
      "label": "Objects",
      "image_count": 9145,
      "indexed": true,
      "indexed_count": 9145,
      "descriptors": [
        {"id": "color", "label": "HSV color", "key": "hsv_grid3_8_4_4", "available": true}
      ]
    }
  ]
}
```

## `GET /catalog`

Browse stored images before choosing one as a query. `dataset` is `objects` or `mirflickr` (the MIRFLICKR-25K collection); `q` filters by image ID or category; `offset` and `limit` page through results.

```text
GET /catalog?dataset=objects&q=butterfly&offset=0&limit=24
```

Each item includes its dataset-relative `id`, display `category`, and API `url`.

## `POST /search`

Request fields:

| Field | Type | Description |
| --- | --- | --- |
| `dataset` | string | `objects` or `mirflickr` (MIRFLICKR-25K) |
| `text` | string, optional | Keywords searched in Elasticsearch |
| `query_image_id` | string, optional | Image ID relative to the selected collection |
| `descriptors` | string array | `color`, `lbp`, `hog`, `cld`, `ehd`, `scd`, `resnet`, or `auto`; availability depends on the selected collection |
| `operator` | string | `AND` filters text candidates before visual reranking; `OR` fuses rankings |
| `k` | integer | Number of results, from 1 to 200 |

Provide `text`, `query_image_id`, or both. Image queries must select at least one descriptor available in that dataset's index.

Example:

```json
{
  "dataset": "objects",
  "text": "butterfly",
  "query_image_id": "butterfly/image_0001.jpg",
  "descriptors": ["color", "lbp", "hog", "cld"],
  "operator": "AND",
  "k": 24
}
```

Responses include `count` and a ranked `results` list. A result has `id`, `url`, `category`, and optional text `score` or visual `distance`.

## `POST /search/upload`

Accepts multipart fields `file`, `dataset`, `descriptors` (comma-separated IDs), `k`, `text`, and `operator`. The upload is processed in memory and is not added to the collection.

## Other endpoints

- `GET /health` reports API and Elasticsearch status.
- `GET /images/{dataset}/{image_id}` serves an image from the selected data folder.
