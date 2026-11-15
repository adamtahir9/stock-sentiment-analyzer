from datetime import datetime, timezone

from sentiment_analyzer.scrapers import FinvizNews

HTML = """<table id="news-table">
<tr><td>Nov-13-26 09:30AM</td><td><a href="https://ex.com/1">Apple beats estimates</a></td></tr>
<tr><td>08:00AM</td><td><a href="https://ex.com/2">Apple to hold event</a></td></tr>
<tr><td>Today 10:15AM</td><td><a href="https://ex.com/3">Apple shares rise</a></td></tr>
</table>"""


def test_finviz_parser_handles_all_date_formats():
    rows = FinvizNews.parse(HTML, "aapl", today=datetime(2026, 11, 15, tzinfo=timezone.utc))
    assert [r["title"] for r in rows] == ["Apple beats estimates", "Apple to hold event", "Apple shares rise"]
    assert rows[1]["published"].strftime("%Y-%m-%d %H:%M") == "2026-11-13 13:00"  # 08:00 ET -> UTC
    assert rows[2]["published"].date().isoformat() == "2026-11-15"
    assert all(r["ticker"] == "AAPL" and r["source"] == "finviz" for r in rows)
