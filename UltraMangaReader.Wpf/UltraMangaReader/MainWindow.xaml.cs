using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Microsoft.Web.WebView2.Core;

namespace UltraMangaReader;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Loaded += OnLoaded;
        InitializeAsync();
    }

    private async void InitializeAsync()
    {
        await ReaderBrowser.EnsureCoreWebView2Async(null);
        NavigateToCurrentUrl();
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        SetStatus("Native reader shell ready. Start aqui e a gente pluga OCR depois.");
    }

    private void BackButton_OnClick(object sender, RoutedEventArgs e)
    {
        if (ReaderBrowser.CanGoBack)
        {
            ReaderBrowser.GoBack();
        }
    }

    private void ForwardButton_OnClick(object sender, RoutedEventArgs e)
    {
        if (ReaderBrowser.CanGoForward)
        {
            ReaderBrowser.GoForward();
        }
    }

    private void ReloadButton_OnClick(object sender, RoutedEventArgs e)
    {
        ReaderBrowser.Reload();
        SetStatus("Page reloaded.");
    }

    private void OpenButton_OnClick(object sender, RoutedEventArgs e)
    {
        NavigateToCurrentUrl();
    }

    private void UrlTextBox_OnKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter)
        {
            NavigateToCurrentUrl();
        }
    }

    private void ReaderBrowser_OnNavigationCompleted(object sender, CoreWebView2NavigationCompletedEventArgs e)
    {
        UrlTextBox.Text = ReaderBrowser.Source?.ToString() ?? UrlTextBox.Text;
        if (e.IsSuccess)
        {
            SetStatus($"Loaded: {UrlTextBox.Text}");
        }
        else
        {
            SetStatus($"Navigation failed: {e.WebErrorStatus}");
        }
    }

    private void StartButton_OnClick(object sender, RoutedEventArgs e)
    {
        var source = (SourceComboBox.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "Auto";
        var translate = (TranslateModeComboBox.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "Echo";
        var analyze = (AnalyzeModeComboBox.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "Manual Analyze";
        SetStatus(
            $"Session armed. source={source} translate={translate} analyze={analyze}. " +
            "Next: wire OCR capture/runtime into this native shell."
        );
    }

    private void InspectButton_OnClick(object sender, RoutedEventArgs e)
    {
        SetStatus("Inspect OCR placeholder. Next build step is viewport capture plus OCR dump.");
    }

    private void SelectRegionButton_OnClick(object sender, RoutedEventArgs e)
    {
        SetStatus("Select Region placeholder. Next build step is draggable viewport selection.");
    }

    private void NavigateToCurrentUrl()
    {
        var raw = UrlTextBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(raw))
        {
            return;
        }

        if (!raw.Contains("://", StringComparison.Ordinal))
        {
            raw = $"https://{raw}";
            UrlTextBox.Text = raw;
        }

        if (Uri.TryCreate(raw, UriKind.Absolute, out var uri))
        {
            if (ReaderBrowser.CoreWebView2 != null)
            {
                ReaderBrowser.CoreWebView2.Navigate(uri.ToString());
            }
            else
            {
                ReaderBrowser.Source = uri;
            }
            SetStatus($"Opening: {uri}");
            return;
        }

        SetStatus("Invalid URL.");
    }

    private void SetStatus(string message)
    {
        StatusTextBlock.Text = message;
    }
}
