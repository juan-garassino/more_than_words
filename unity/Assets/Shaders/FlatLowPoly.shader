// Living Tales — Flat Low-Poly Shader
// Flat shading with vertex colors, subtle rim light, and warm ambient.
// No textures needed — all color from vertex colors or material tint.

Shader "LivingTales/FlatLowPoly"
{
    Properties
    {
        _Color ("Tint Color", Color) = (1, 1, 1, 1)
        _RimColor ("Rim Color", Color) = (1, 0.95, 0.85, 1)
        _RimPower ("Rim Power", Range(0.5, 8.0)) = 3.0
        _RimStrength ("Rim Strength", Range(0, 1)) = 0.3
        _AmbientColor ("Ambient Color", Color) = (0.85, 0.82, 0.78, 1)
        _AmbientStrength ("Ambient Strength", Range(0, 1)) = 0.3
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 100

        Pass
        {
            Tags { "LightMode"="UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fog

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float4 color : COLOR;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 normalWS : TEXCOORD0;
                float3 viewDirWS : TEXCOORD1;
                float4 vertexColor : COLOR;
                float fogFactor : TEXCOORD2;
            };

            CBUFFER_START(UnityPerMaterial)
                float4 _Color;
                float4 _RimColor;
                float _RimPower;
                float _RimStrength;
                float4 _AmbientColor;
                float _AmbientStrength;
            CBUFFER_END

            Varyings vert(Attributes input)
            {
                Varyings output;

                VertexPositionInputs posInputs = GetVertexPositionInputs(input.positionOS.xyz);
                VertexNormalInputs normalInputs = GetVertexNormalInputs(input.normalOS);

                output.positionCS = posInputs.positionCS;
                output.normalWS = normalInputs.normalWS;
                output.viewDirWS = GetWorldSpaceNormalizeViewDir(posInputs.positionWS);
                output.vertexColor = input.color;
                output.fogFactor = ComputeFogFactor(posInputs.positionCS.z);

                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                // Flat normal (no interpolation — use face normal)
                float3 normal = normalize(input.normalWS);

                // Main light
                Light mainLight = GetMainLight();
                float NdotL = saturate(dot(normal, mainLight.direction));

                // Base color from vertex color * tint
                float3 baseColor = input.vertexColor.rgb * _Color.rgb;

                // Diffuse lighting (half-lambert for softer look)
                float halfLambert = NdotL * 0.5 + 0.5;
                float3 diffuse = baseColor * halfLambert * mainLight.color.rgb;

                // Ambient
                float3 ambient = baseColor * _AmbientColor.rgb * _AmbientStrength;

                // Rim light
                float rim = 1.0 - saturate(dot(normalize(input.viewDirWS), normal));
                rim = pow(rim, _RimPower);
                float3 rimColor = _RimColor.rgb * rim * _RimStrength;

                // Combine
                float3 finalColor = diffuse + ambient + rimColor;

                // Fog
                finalColor = MixFog(finalColor, input.fogFactor);

                return half4(finalColor, 1.0);
            }
            ENDHLSL
        }

        // Shadow caster pass
        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode"="ShadowCaster" }

            ZWrite On
            ZTest LEqual

            HLSLPROGRAM
            #pragma vertex ShadowVert
            #pragma fragment ShadowFrag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"

            struct ShadowAttributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
            };

            struct ShadowVaryings
            {
                float4 positionCS : SV_POSITION;
            };

            float3 _LightDirection;

            ShadowVaryings ShadowVert(ShadowAttributes input)
            {
                ShadowVaryings output;
                float3 posWS = TransformObjectToWorld(input.positionOS.xyz);
                float3 normalWS = TransformObjectToWorldNormal(input.normalOS);
                posWS = ApplyShadowBias(posWS, normalWS, _LightDirection);
                output.positionCS = TransformWorldToHClip(posWS);
                return output;
            }

            half4 ShadowFrag(ShadowVaryings input) : SV_Target
            {
                return 0;
            }
            ENDHLSL
        }
    }

    FallBack "Universal Render Pipeline/Lit"
}
