import typescriptEslint from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";
import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default [
    eslint.configs.recommended,               // ESLint recommended rules
    ...tseslint.configs.recommendedTypeChecked,

    {
    files: ["**/*.ts"],
    plugins: {
        "@typescript-eslint": typescriptEslint,
    },

    languageOptions: {
        parser: tsParser,
        ecmaVersion: 2022,
        sourceType: "module",
        parserOptions: {
            projectService: true,
            tsconfigRootDir: import.meta.dirname,
          },    
    },
    rules: {
        
        "@typescript-eslint/naming-convention": ["warn", {
            selector: "import",
            format: ["camelCase", "PascalCase"],
        }],

        curly: ["warn", "multi-line"],
        eqeqeq: "warn",
        "no-throw-literal": "warn",
        semi: "warn",
    },
}
];