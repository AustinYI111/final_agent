import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data import DataProvider


def test_data_provider_basic():
    provider = DataProvider(
        symbol="000001",
        start_date="20200101",
        end_date="20251231"
    )

    # 强制重新获取数据（不使用缓存）
    df = provider.fetch_data(use_cache=False)
    df = provider.add_indicators(df)
    
    print(f"\n数据总行数: {len(df)}")
    print(f"日期范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"\n后5行数据（最新）:")
    print(df.tail())

    assert "close" in df.columns
    assert "ma_5" in df.columns
    assert "rsi" in df.columns


if __name__ == "__main__":
    test_data_provider_basic()
