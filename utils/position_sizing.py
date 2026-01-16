def calculate_position(balance, price, percentage=0.2):
    capital = balance * percentage
    shares = int(capital // price)
    return shares
