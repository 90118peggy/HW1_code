"""舊版相容入口；正式檢查器已移至 data_pipeline.inspect_dataset。"""

from data_pipeline.inspect_dataset import (
    LABELS, REQUIRED_COLUMNS, SPLITS, inspect_dataset, main, print_summary,
)

if __name__ == "__main__":
    raise SystemExit(main())
