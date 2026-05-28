import sys
print("Executable:", sys.executable)
print("Path:", sys.path)
try:
    import tabulate
    print("tabulate imported successfully in check script:", tabulate.__file__)
except Exception as e:
    print("tabulate import failed in check script:", e)
