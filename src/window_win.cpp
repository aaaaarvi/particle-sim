#include <iostream>

#include "window_win.h"

LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_CLOSE:
        DestroyWindow(hWnd);
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }

    return DefWindowProc(hWnd, uMsg, wParam, lParam);
}

MyWindow::MyWindow(int width, int height, int offset_w, int offset_h)
    : m_width(width), m_height(height),
      m_offset_w(offset_w), m_offset_h(offset_h),
      m_hInstance(GetModuleHandle(nullptr))
{
    const char* CLASS_NAME = "My Window Class";

    WNDCLASS wndClass = {};
    wndClass.style = CS_HREDRAW | CS_VREDRAW;
    wndClass.lpszClassName = CLASS_NAME;
    wndClass.hInstance = m_hInstance;
    wndClass.hIcon = LoadIcon(NULL, IDI_WINLOGO);
    wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    wndClass.lpfnWndProc = WindowProc;

    RegisterClass(&wndClass);

    DWORD style = WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU;

    RECT rect;
    rect.left = m_offset_w;
    rect.top = m_offset_h;
    rect.right = rect.left + m_width;
    rect.bottom = rect.top + m_height;

    AdjustWindowRect(&rect, style, false); // correct rect dimensions to be the inner dimensions

    m_hWnd = CreateWindowEx(
        0,
        CLASS_NAME,
        "Title",
        style,
        rect.left,
        rect.top,
        rect.right - rect.left,
        rect.bottom - rect.top,
        NULL,
        NULL,
        m_hInstance,
        NULL
    );

    ShowWindow(m_hWnd, SW_SHOW);
}

MyWindow::~MyWindow()
{
    const char* CLASS_NAME = "My Window Class";

    UnregisterClass(CLASS_NAME, m_hInstance);
}

bool MyWindow::ProcessMessages()
{
    MSG msg = {};

    while (PeekMessage(&msg, nullptr, 0u, 0u, PM_REMOVE))
    {
        if (msg.message == WM_QUIT)
        {
            return false;
        }

        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return true;
}

void MyWindow::DrawPixels(std::vector<std::vector<int>> pixels)
{
    RECT r;
    GetClientRect(m_hWnd, &r);

    if (r.bottom == 0) {
        return;
    }

    HDC hdc = GetDC(m_hWnd);
    if (hdc == NULL) {
        return;
    }

    SelectObject(hdc, GetStockObject(BLACK_PEN));
    SelectObject(hdc, GetStockObject(BLACK_BRUSH));
    Rectangle(hdc, 0, 0, r.right, r.bottom);

    for (unsigned int i = 0; i < pixels.size(); i++) {
        SetPixel(hdc, pixels[i][0], pixels[i][1], RGB(255, 255, 255));
    }

    ReleaseDC(m_hWnd, hdc);
}
