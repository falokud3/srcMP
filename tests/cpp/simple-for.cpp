
void test() {
    int z = 0;
    for (int i = 0; i < 10; i++) {
        int x = i;
        int y = x + i;
        z = y + 2;
    }

    for (int i = 0; i < 10; i++) {
        int x = i;
        int y = i;
        z = i;
    }
}