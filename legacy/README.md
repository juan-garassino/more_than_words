# More Than Words (formerly Symbolic Tamagotchi AI)

## Data & Deployment Guide

> **🌟 Project Vision**: This project is evolving from a virtual pet into a **Generative Quest Game**. 
> Read [VISION.md](VISION.md) to understand how we are leveraging the Symbolic Transformer engine for procedural storytelling, NPC brains, and more.

### Prerequisites
- Mac with macOS Ventura or later
- Xcode 14 or later
- iOS Device (iPhone) running iOS 16.0+
- Apple ID (free) or Apple Developer Account

### 1. Open the Project
1. Navigate to this folder in Finder.
2. If you haven't already, create an Xcode project:
   - Open Xcode > **Create a new Xcode project**.
   - Select **iOS** > **App**.
   - Product Name: `SymbolicTamagotchiAI`.
   - Interface: **SwiftUI**.
   - Language: **Swift**.
   - Storage: **Core Data** (Check this box).
   - Create it in the parent folder, overwriting the placeholder files if asked, or drag the generated Swift files from this folder into the new Xcode project.
   - **Crucial**: Ensure `SymbolicTamagotchiAIApp.swift`, `ContentView.swift`, and all files in `Models/`, `ViewModels/`, `Services/`, `Views/`, `Resources/` are present in the Xcode project navigator.

### 2. Add the Core ML Model
1. Locate your `TamagotchiTransformer.mlpackage` file.
2. Drag and drop it into the Project Navigator (left sidebar) in Xcode.
3. Ensure "Copy items if needed" is checked.
4. Verify `ModelHandler.swift` logic maps to the generated class (see comments in that file).

### 3. Configure Signing
1. Click the **SymbolicTamagotchiAI** project icon at the top of the Project Navigator.
2. Select the **Signing & Capabilities** tab.
3. Under **Team**, select your Apple ID (Personal Team).
   - If none is listed, click **Add an Account...** and log in.
4. Ensure **Bundle Identifier** is unique (e.g., `com.yourname.SymbolicTamagotchiAI`).

### 4. Deploy to iPhone
1. Connect your iPhone to your Mac via USB/Lightning/USB-C.
2. Unlock your iPhone.
3. In Xcode, look at the top toolbar. Click the device selector (likely says "iPhone 15 Pro" or "Any iOS Device").
4. Select your connected iPhone from the list.
5. Press **Cmd + R** or click the **Play** button to build and run.

### 5. Troubleshooting "Untrusted Developer"
If the app installs but won't open on your phone:
1. On your iPhone, go to **Settings** > **General** > **VPN & Device Management** (or **Device Management**).
2. Tap your Apple ID email under "Developer App".
3. Tap **Trust [Your Email]**.
4. Launch the app again.

### 6. Persistence Note
The app uses Core Data / UserDefaults. If you delete the app from your phone, your pet's stats and history will be reset.
