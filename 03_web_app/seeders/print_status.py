def print_status(table_name, counter):
    status = "✅" if counter > 0 else "❎"
    adding = f"adding {counter}" if counter > 0 else "no additions"
    print(f"{status} {table_name}: {adding}")