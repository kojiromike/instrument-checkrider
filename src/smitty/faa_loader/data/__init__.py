import csv
import importlib.resources

with (
    importlib.resources.path(__package__, "sources.csv") as csv_path,
    csv_path.open("r") as f,
):
    dr = csv.DictReader(f)
    SOURCES = {
        r["Reference Number"]: r["Url"]
        for r in dr
        if r["Reference Number"] != "DA-40 XML AFM"
    }

__all__ = ("SOURCES",)
