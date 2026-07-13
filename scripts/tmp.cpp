#include <chrono>
#include <iostream>

int c[100];

int main() {
    auto t = std::chrono::high_resolution_clock::now();
    int cnt = 0;
    auto start = std::chrono::high_resolution_clock::now();
    int total = 0;
    int sum = 0;
    while(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count() < 1000) {
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - t);
        t = std::chrono::high_resolution_clock::now();
        int elapsed = duration.count();
        elapsed = std::min(elapsed, 9999);
        c[elapsed / 100] += 1;
        if(elapsed > 100) total += elapsed;
        sum += elapsed; 
        ++cnt;
    }
    for(int i = 0; i < 100; ++i) {
        printf("%d %.10lf %d\n", i, c[i] * 1.0 / cnt, c[i]);
    }
    printf("%d/%d\n", total, sum);
    return 0;
}