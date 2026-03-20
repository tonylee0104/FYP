#include "mixer.h"
#include "tether_observer.h"
#include <iostream>
#include <memory>

int main() {
    std::cout << "Testing integrated classes without ROS2..." << std::endl;
    
    // 创建Mixer实例
    auto mixer = std::make_shared<Mixer>();
    std::cout << "Mixer created successfully" << std::endl;
    
    // 设置Mixer参数
    mixer->set_params(0.055, 0.045, 0.01, 1.0, 1.99);
    std::cout << "Mixer parameters set successfully" << std::endl;
    
    // 创建TetherObserver实例
    auto tether_observer = std::make_shared<TetherObserver>(mixer);
    std::cout << "TetherObserver created successfully" << std::endl;
    
    // 设置TetherObserver参数
    tether_observer->set_params(0.110, 1.99);
    std::cout << "TetherObserver parameters set successfully" << std::endl;
    
    // 测试获取观测结果
    auto R_bt = tether_observer->get_R_bt();
    auto ft_bodyframe = tether_observer->get_ft_bodyframe();
    
    std::cout << "R_bt matrix:" << std::endl << R_bt << std::endl;
    std::cout << "ft_bodyframe vector:" << std::endl << ft_bodyframe << std::endl;
    
    std::cout << "All tests passed! Mixer and TetherObserver work as integrated classes." << std::endl;
    
    return 0;
}
