class A:
    def __iter__(self):
        self.a=1
        return self
    def __next__(self):
        x=self.a
        self.a+=1
        return x

aa=A()
aa_iter=iter(aa)

print(next(aa_iter))