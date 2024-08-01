class Account:
    def __init__(self, cash: int | float):
        self.balance = cash

    def __str__(self):
        return f'{self.balance}'

    def __repr__(self):
        return f'Account(balance == {self.balance})'

    def deposit(self, amount: int | float) -> None:
        if amount < 0:
            raise ValueError('Cannot deposit with negative amount')

        old_balance = self.balance
        self.balance += amount
        print(f'Current balance: {old_balance} + {amount} = {self.balance}')

    def withdraw(self, amount: int | float) -> None:
        if amount > self.balance:
            print(f'Withdraw amount: ${amount}, Current Balance: ${self.balance}')
            raise ValueError('Cannot withdraw with amount greater than the balance')
        elif amount < 0:
            raise ValueError('Cannot withdraw with negative amount')

        old_balance = self.balance
        self.balance -= amount
        print(f'Current balance: {old_balance} - {amount} = {self.balance}')
