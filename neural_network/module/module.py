class Module():
    def __init__(self) -> None:
        pass
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        NotImplementedError("Implement self forward method")

    def grad(self, x):
        NotImplementedError("Implement self grad method")