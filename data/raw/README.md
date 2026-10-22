# Raw snapshots

`collect` writes immutable OpenAQ response pages under a named snapshot directory.
Do not edit downloaded JSON. Preserve a completed snapshot with its
`request_manifest.csv` using Git LFS or external versioned storage.

Use a different snapshot name for each station cohort, for example
`core5-2019-2024-v1` and `gth10-2019-2024-v1`. A PM2.5 snapshot is complete only
when `request_manifest.csv` exists; weather is complete only when
`weather_request_manifest.csv` exists.

