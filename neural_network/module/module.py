class Module():
    def __init__(self) -> None:
        self.no_grad = False
        self.last_x = None
    
    def __call__(self, x):
        self.last_x = x if not self.no_grad else self.last_x
        return self.forward(x)
    
    def forward(self, x):
        NotImplementedError("Implement self forward method")

    def grad(self, x):
        NotImplementedError("Implement self grad method")