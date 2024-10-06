# Function to convert Fahrenheit to Celsius


def fahrenheit_to_celsius(fahrenheit):
    celsius = (5 / 9) * (fahrenheit - 32)
    return celsius


fahrenheit = float(input("Enter the temperature in Fahrenheit: "))

# Call the conversion function
celsius = fahrenheit_to_celsius(fahrenheit)

# Output the Celsius temperature
print(f"{fahrenheit} Fahrenheit is equal to {celsius:.2f} Celsius")

##Updated 09/10/2024
