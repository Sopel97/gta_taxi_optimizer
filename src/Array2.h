#pragma once

#include <algorithm>
#include <memory>

namespace util
{
    template <typename T>
    struct Array2
    {
        Array2(int width, int height) :
            m_data(std::make_unique<T[]>(width * height)),
            m_width(width),
            m_height(height)
        {

        }

        Array2(int width, int height, const T& v) :
            m_data(std::make_unique<T[]>(width * height)),
            m_width(width),
            m_height(height)
        {
            std::fill(m_data.get(), m_data.get() + width * height, v);
        }

        [[nodiscard]] const T& operator()(int x, int y) const
        {
            return m_data[x * m_height + y];
        }

        [[nodiscard]] T& operator()(int x, int y)
        {
            return m_data[x * m_height + y];
        }

        [[nodiscard]] int width() const
        {
            return m_width;
        }

        [[nodiscard]] int height() const
        {
            return m_height;
        }

    private:
        std::unique_ptr<T[]> m_data;
        int m_width;
        int m_height;
    };
}
