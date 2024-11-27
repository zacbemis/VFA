import yfinance as yf


def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info


def calculate_pe_ratio(info):
    try:
        pe_ratio = info.get("trailingPE", None)
        if pe_ratio:
            return pe_ratio
        else:
            print("P/E ratio data is not available.")
    except KeyError:
        print("P/E ratio is unavailable.")
    return None


def calculate_pb_ratio(info):
    try:
        pb_ratio = info.get("priceToBook", None)
        if pb_ratio:
            return pb_ratio
        else:
            print("P/B ratio data is not available.")
    except KeyError:
        print("P/B ratio is unavailable.")
    return None


def evaluate_stock(pe_ratio, pb_ratio, sector_avg_pe=15, sector_avg_pb=1.5):
    valuation = ""

    if pe_ratio and pe_ratio < sector_avg_pe:
        valuation += "The stock might be undervalued based on the P/E ratio.\n"
    elif pe_ratio and pe_ratio > sector_avg_pe:
        valuation += "The stock might be overvalued based on the P/E ratio.\n"

    if pb_ratio and pb_ratio < sector_avg_pb:
        valuation += "The stock might be undervalued based on the P/B ratio.\n"
    elif pb_ratio and pb_ratio > sector_avg_pb:
        valuation += "The stock might be overvalued based on the P/B ratio.\n"

    return valuation if valuation else "Insufficient data for valuation."


def main():
    ticker = input("Enter the stock ticker (e.g., AAPL, TSLA): ").upper()
    info = fetch_stock_data(ticker)

    pe_ratio = calculate_pe_ratio(info)
    pb_ratio = calculate_pb_ratio(info)

    print("\nStock Analysis:")
    if pe_ratio:
        print(f"P/E Ratio: {pe_ratio}")
    if pb_ratio:
        print(f"P/B Ratio: {pb_ratio}")

    valuation = evaluate_stock(pe_ratio, pb_ratio)
    print("\nValuation Analysis:")
    print(valuation)


if __name__ == "__main__":
    main()
