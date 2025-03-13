def is_prime(number):
    """Check if a number is a prime number."""
    if number <= 1:
        return False
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False
    return True

def main():
    """Main function to test prime numbers."""
    test_numbers = [2, 3, 4, 5, 10, 13, 17, 19, 23, 24, 29]
    for num in test_numbers:
        if is_prime(num):
            print(f"{num} is a prime number.")
        else:
            print(f"{num} is not a prime number.")

if __name__ == "__main__":
    main()
    print("ganesh")