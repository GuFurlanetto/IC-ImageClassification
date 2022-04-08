lista1 = [1, 2, 3]

lista2 = [4, 5, 6, 7, 8, 9, 10, 11, 12]


if __name__ == '__main__':
    for i in range(len(lista2)):
        print(lista1[i%3])
        print(lista2[i])
