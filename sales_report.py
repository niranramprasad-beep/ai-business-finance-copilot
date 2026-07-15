sales = [12500, 14200, 11800, 15600, 16100, 13900, 17250, 18400, 16800, 19100, 17900, 21500]
def profit_margin(revenue, cost):
    profit=revenue-cost
    return profit/revenue
print("Profit Margin:", profit_margin(500,300))
print("Highest Month Sales amount:", max(sales))
print("Lowest Month Sales amount:", min(sales))