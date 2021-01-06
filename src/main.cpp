#include <Optimize.hpp>
#include <iostream>

int main()
{
    //f = 2|2x1-2x2|+(2x1-1)(2x2-1)
    Optimize<2>::funk_type funk = [](Optimize<2>::vars_type x){
        return 2*fabs(2*x(0)-2*x(1))+(2*x(0)-1)*(2*x(1)-1);
    };

    Optimize<2> optimizer(funk, Optimize<2>::vars_type::Zero(),Optimize<2>::vars_type::Ones());
    optimizer.setNumObservations(125);
    optimizer.setMaxDepth(10);
    optimizer.setMaxIterations(25);
    optimizer.setThreshold(3);

    Optimize<2>::vars_type minimum = optimizer.run();
    std::cout << "Found Minimum: " <<std::endl;
    std::cout<< minimum << std::endl;
    std::cout << "f at minimum: " << funk(minimum) << std::endl;

    return 0;
}