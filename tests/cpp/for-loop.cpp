#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

vector<double> calcSin(vector<int> inputs) {
    vector<double> outputs;
    for (int i = 0; i < inputs.size(); i++) {
        outputs.push_back(sin(inputs.at(i)));
    }
    return outputs;
}

vector<double> nonParallelCalcSin(vector<int> inputs) {
    vector<double> outputs;
    for (int i = 0; i < inputs.size(); i++) {
        outputs.push_back(sin(inputs.at(i + 1)));
    }
    return outputs;
}

//int main() {
//    vector<int> inputs = {0, 10, 20, 30};
//    vector<double> outputs = calcSin(inputs);
//    for (double num: outputs) {
//        cout << num << endl;
//    }
//}