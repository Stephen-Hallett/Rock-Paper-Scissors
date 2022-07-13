from RPS import RPS

long = {'r': 'Rock',
        'p': 'Paper',
        's': 'Scissors'}

name = input("What is your name? ")
rps = RPS(name)

ans = []
while True:
    try:
        prediction = rps.predict()
        choice = input("Rock Paper Scissors! ")
        print()
        if choice == "q":
            break
        print("I pick {}!".format(long[prediction]))
        rps.played(choice,prediction)
        print()
        print(rps)
        print()
    except Exception as e:
        print(e)
        pass

rps.quit()
