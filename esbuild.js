import esbuild from "esbuild"

const production = process.argv.includes('--production');
const watch = process.argv.includes('--watch');
const extension = process.argv.includes('--extension');

/**
 * @type {import('esbuild').Plugin}
 */
const esbuildProblemMatcherPlugin = {
	name: 'esbuild-problem-matcher',

	setup(build) {
		build.onStart(() => {
			console.log('[watch] build started');
		});
		build.onEnd((result) => {
			result.errors.forEach(({ text, location }) => {
				console.error(`✘ [ERROR] ${text}`);
				console.error(`    ${location.file}:${location.line}:${location.column}:`);
			});
			console.log('[watch] build finished');
		});
	},
};

async function main() {
	const ctx = await esbuild.context({
		entryPoints: extension ? ['src/ext/extension.ts'] : [
			'src/srcMP/srcmp.ts',
			'src/srcSim/srcsim.ts',	
		],
		bundle: true,
		format: 'cjs',
		outExtension: extension ? {} : { '.js': '.cjs' },
		minify: production,
		sourcemap: !production,
		sourcesContent: false,
		platform: 'node',
		outdir: extension ? './build/ext' : './build',
		external: ['vscode'], // excludes code from bundle
		logLevel: 'silent',
		plugins: [
			/* add to the end of plugins array */
			esbuildProblemMatcherPlugin,
		],
		color: true,
	});
	if (watch) {
		await ctx.watch();
	} else {
		await ctx.rebuild();
		await ctx.dispose();
	}
}

main().catch(e => {
	console.error(e);
	process.exit(1);
});
